//! Cross-thread handle to a [`VirtualMachine`].
//!
//! Worker VMs are freed when the worker terminates (`WebWorker::shutdown`),
//! so any thread that stored a raw `&VirtualMachine` (HTTP thread, work pool,
//! addon threads) can dereference freed memory. `VMHandle` is the required
//! currency for such code: the VM allocation may only be touched inside
//! [`VMHandle::with`], which pins it via a gate that worker shutdown closes
//! (and drains) before the dealloc.

use core::ptr::NonNull;
use std::sync::Arc;

use bun_threading::ShutdownGate;

use crate::virtual_machine::VirtualMachine;

#[derive(Clone)]
pub struct VMHandle {
    vm: NonNull<VirtualMachine>,
    gate: Arc<ShutdownGate>,
}

// SAFETY: the raw VM pointer is only dereferenced inside `with()` while the
// gate is held open; teardown close_and_wait()s the gate before the
// allocation is freed.
unsafe impl Send for VMHandle {}
// SAFETY: same gate protocol as `Send`; `VirtualMachine`'s single-JS-thread
// invariant covers all `&VirtualMachine` uses reachable from `with()`.
unsafe impl Sync for VMHandle {}

impl VMHandle {
    pub(crate) fn new(vm: NonNull<VirtualMachine>, gate: Arc<ShutdownGate>) -> Self {
        Self { vm, gate }
    }

    /// Run `f` against the VM if it is still alive. Returns `None` once the
    /// VM has been torn down (worker terminated) — the memory is gone and the
    /// caller must take its no-VM cleanup path.
    ///
    /// `f` runs with the VM teardown blocked on it; it must not block on the
    /// VM's own JS thread (enqueue + wakeup style work only).
    pub fn with<R>(&self, f: impl FnOnce(&VirtualMachine) -> R) -> Option<R> {
        if !self.gate.enter() {
            return None;
        }
        // SAFETY: gate held open — `close_and_wait()` in worker shutdown
        // blocks until we `leave()`, so the allocation outlives this call.
        let result = f(unsafe { self.vm.as_ref() });
        self.gate.leave();
        Some(result)
    }

    /// Enqueue a concurrent task on the VM's JS-thread event loop. Returns
    /// `false` if the VM is already torn down — `make_task` then never runs
    /// and the caller must free whatever it would have wrapped via its
    /// [`DisposeAfterVmDestroyed`] path (JSC handle fields are forgotten —
    /// their slot storage died with the VM — everything else is dropped).
    /// On success the queue takes ownership of the produced
    /// `ConcurrentTask` via its intrusive `next` link. Enqueueing into a live
    /// but shutting-down VM still succeeds — worker teardown drains the queue
    /// after closing the gate; on the main VM the queue outlives the process.
    /// Callers that must not enqueue then (fetch) check `is_shutting_down`
    /// inside [`Self::with`] themselves.
    pub fn enqueue_task_concurrent(
        &self,
        make_task: impl FnOnce() -> NonNull<crate::event_loop::ConcurrentTaskItem>,
    ) -> bool {
        self.with(|vm| vm.event_loop_shared().enqueue_task_concurrent(make_task()))
            .is_some()
    }

    /// [`Self::enqueue_task_concurrent`] for the common shape where the task
    /// is `owner`'s inline intrusive `concurrent_task` field: on `false` the
    /// field is left untouched and `owner` is freed via
    /// [`DisposeAfterVmDestroyed`].
    ///
    /// `owner` must be live and exclusively owned by the caller (the same
    /// contract every producer already upholds for the enqueue itself).
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn enqueue_intrusive<T: bun_event_loop::Taskable + DisposeAfterVmDestroyed>(
        &self,
        ct: &mut crate::event_loop::ConcurrentTaskItem,
        owner: *mut T,
    ) -> bool {
        use bun_event_loop::ConcurrentTask::AutoDeinit;
        let queued = self
            .enqueue_task_concurrent(|| NonNull::from(ct.from(owner, AutoDeinit::ManualDeinit)));
        if !queued {
            // SAFETY: `false` ⇒ the queue never took ownership; the completing
            // thread is the sole owner of `owner`, and the owning VM is gone.
            unsafe { T::dispose_after_vm_destroyed(owner) };
        }
        queued
    }

    /// Pin the VM allocation (and everything owned by its worker thread's
    /// shutdown, e.g. the env loader) for a bounded external operation such
    /// as a bundle run. `None` if the VM is gone. Worker terminate blocks
    /// until the returned guard drops, so only bounded work may hold one; the
    /// guard owns its own `Arc` and may outlive `self`.
    #[must_use]
    pub fn pin_for_bounded_work(&self) -> Option<bun_threading::GateGuest> {
        bun_threading::GateGuest::enter(&self.gate)
    }

    /// The VM, without the gate. Only callable on the VM's own JS thread,
    /// where teardown cannot run concurrently. The `'static` lifetime mirrors
    /// the thread-lifetime guarantee callers already relied on.
    pub fn vm(&self) -> &'static VirtualMachine {
        debug_assert!(
            VirtualMachine::get_or_null() == Some(self.vm.as_ptr()),
            "VMHandle::vm() called off the VM's JS thread"
        );
        // SAFETY: same-thread access; the VM is freed only by this thread,
        // after it stops running JS (worker shutdown / process exit).
        unsafe { self.vm.as_ref() }
    }
}

/// Free a completion object whose owning VM has been destroyed.
///
/// The destroyed VM's JSC heap, HandleSet and WeakBlocks are already freed
/// wholesale, so JSC handle wrappers (`Strong`, `StrongOptional`,
/// `JSPromiseStrong`, `Weak`) are plain pointers into dead memory:
/// `core::mem::forget(core::mem::take(&mut field))` releases them without a
/// byte of leak — the slot storage was owned by (and died with) the VM.
/// Everything else the object owns (boxes, buffers, fds, process-heap
/// natives whose destructors don't touch the JSC heap) must be freed
/// normally. Runs on whatever thread discovered the dead VM (work pool,
/// HTTP thread); implementations must not touch the VM, its loop, or any
/// thread-local of the dead thread.
///
/// # Safety
/// Implementations free `this`; callers must own it exclusively.
pub unsafe trait DisposeAfterVmDestroyed {
    /// # Safety
    /// `this` is live, exclusively owned by the caller, and its owning VM has
    /// been destroyed (its gate is closed).
    unsafe fn dispose_after_vm_destroyed(this: *mut Self)
    where
        Self: Sized,
    {
        // SAFETY: forwarded caller contract.
        unsafe { dispose_box_for_dead_vm(this) };
    }
}

/// Free a heap object inside a [dead-VM disposal
/// scope](bun_core::dead_vm_scope): JSC handle wrappers, protect pins, loop
/// refs and env refs no-op their release (their storage died with the VM),
/// while every other field drops normally. Types owning resources with no
/// drop glue (raw fds, `Buffer`, `BunString`, C contexts) must override
/// [`DisposeAfterVmDestroyed::dispose_after_vm_destroyed`] and release those
/// explicitly before (or instead of) calling this.
///
/// # Safety
/// `this` is a live `heap::alloc`/`Box` allocation exclusively owned by the
/// caller; its owning VM has been destroyed (or is past teardown).
pub unsafe fn dispose_box_for_dead_vm<T>(this: *mut T) {
    let _scope = bun_core::dead_vm_scope::DeadVmDisposalScope::enter();
    // SAFETY: caller contract.
    drop(unsafe { bun_core::heap::take(this) });
}
