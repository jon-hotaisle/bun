//! Dead-VM disposal scope.
//!
//! When a completion object outlives its worker VM, its JSC handle wrappers
//! (`Strong`, `Weak`, protect pins, …) point into the VM's freed HandleSet
//! and must be forgotten rather than released — their slot storage was owned
//! by (and died with) the VM, so forgetting leaks nothing. Instead of every
//! disposal site hand-splitting fields, disposal runs inside this thread-
//! local scope and the handle types' `Drop`/`deinit` no-op while it is
//! active, so a plain `drop(Box)` frees everything else normally.

use core::cell::Cell;

std::thread_local! {
    static IN_DEAD_VM_DISPOSAL: Cell<bool> = const { Cell::new(false) };
}

/// True while the current thread is inside a [`DeadVmDisposalScope`]:
/// releases that would touch the dead VM (handle slots, loop refs, protect
/// pins) must be skipped.
#[inline]
pub fn in_dead_vm_disposal() -> bool {
    IN_DEAD_VM_DISPOSAL.with(Cell::get)
}

/// RAII guard entering the dead-VM disposal scope on this thread.
/// Not `Send`; nesting is allowed (the outermost guard clears the flag).
pub struct DeadVmDisposalScope {
    was_active: bool,
    _not_send: core::marker::PhantomData<*mut ()>,
}

impl DeadVmDisposalScope {
    #[must_use]
    pub fn enter() -> Self {
        let was_active = IN_DEAD_VM_DISPOSAL.with(|f| f.replace(true));
        Self {
            was_active,
            _not_send: core::marker::PhantomData,
        }
    }
}

impl Drop for DeadVmDisposalScope {
    fn drop(&mut self) {
        IN_DEAD_VM_DISPOSAL.with(|f| f.set(self.was_active));
    }
}
