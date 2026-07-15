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
    /// `false` if the VM is already torn down — unreachable for producers
    /// holding a [`Self::pin`] (the required protocol: worker terminate
    /// waits for all pins before tearing down).
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

    /// Enqueue while the caller holds a [`Self::pin`]: infallible — the
    /// guest keeps the VM allocation alive even while `close_and_wait` is
    /// draining (the gate refuses NEW entries then, which is exactly why
    /// pinned producers must enqueue through their own guest).
    pub fn enqueue_task_concurrent_pinned(
        &self,
        _pin: &bun_threading::GateGuest,
        make_task: impl FnOnce() -> NonNull<crate::event_loop::ConcurrentTaskItem>,
    ) {
        // SAFETY: the caller's guest pins the allocation; the queue outlives
        // the gate close (drained by worker shutdown before the dealloc).
        unsafe { self.vm.as_ref() }
            .event_loop_shared()
            .enqueue_task_concurrent(make_task());
    }

    /// [`Self::enqueue_task_concurrent_pinned`] for the intrusive
    /// `concurrent_task`-field shape.
    pub fn enqueue_intrusive_pinned<T: bun_event_loop::Taskable>(
        &self,
        pin: &bun_threading::GateGuest,
        ct: &mut crate::event_loop::ConcurrentTaskItem,
        owner: *mut T,
    ) {
        use bun_event_loop::ConcurrentTask::AutoDeinit;
        self.enqueue_task_concurrent_pinned(pin, || {
            NonNull::from(ct.from(owner, AutoDeinit::ManualDeinit))
        });
    }

    /// Pin the VM allocation for the duration of an async operation: worker
    /// terminate blocks in `close_and_wait` until every pin drops, so a
    /// pinned producer's completion enqueue cannot lose the race with
    /// teardown. Hold from job creation until right AFTER the completion is
    /// enqueued (never until the JS thread consumes it — the JS thread is
    /// the one waiting). `None` if the VM is already gone; the guard owns
    /// its own `Arc` and may outlive `self`.
    #[must_use]
    pub fn pin_for_bounded_work(&self) -> Option<bun_threading::GateGuest> {
        bun_threading::GateGuest::enter(&self.gate)
    }

    /// [`Self::pin_for_bounded_work`] for creation sites on the VM's own JS
    /// thread, where the VM is alive by construction.
    #[must_use]
    pub fn pin(&self) -> bun_threading::GateGuest {
        self.pin_for_bounded_work()
            .expect("VMHandle::pin on a live VM (JS-thread creation site)")
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
