package homesystem;

/**
 * A Lockable interface for devices.
 * A locked device should not be able to be turned on.
 */
public interface Lockable {

    /**
     * Sets the lock status to locked
     */
    void lock();

    /**
     * Sets the lock status to unlocked
     */
    void unlock();

    /**
     * Queries the lock status
     * @return true if locked, false otherwise
     */
    boolean isLocked();

    /**
     * Toggles the lock status. If locked, unlock the device, and vice-versa
     * @return the lock status, true if locked, false otherwise
     */
    boolean toggleLock();
}
