package homesystem;

/**
 * A class to represent cameras in the home automation system
 */
public class Camera extends Device implements Lockable{

    private boolean locked;

    /**
     * Constructor
     * @param n the initial camera name
     * @param i the camera id
     */
    public Camera(String n, String i){
        super(n, i);
        locked = true;
    }

    /**
     * Constructor with default camera name
     * @param i the camera id
     */
    public Camera(String i){
        this("Camera", i);
    }

    /**
     * Query for the camera's status
     * @return "on/off, locked/unlocked"
     */
    public String getStatus(){
        String lock = isLocked() ? "locked" : "unlocked";
        if(this.isOn()) return "on, " + lock;
        return "off, " + lock;
    }

    /**
     * Query for the String representation of the camera: "Camera id: name (status)"
     * @return the String representation of the camera
     */
    @Override
    public String toString(){ return "Camera " + super.toString() + "(" + this.getStatus() + ")"; }

    /**
     * Command to lock the camera <br> Also turns it off
     */
    public void lock(){ locked = true; turnOff(); }

    /**
     * Command to unlock the camera
     */
    public void unlock(){ locked = false; }

    /**
     * Query to check the lock status
     * @return true if locked, false otherwise
     */
    public boolean isLocked(){ return locked; }

    /**
     * Command to toggle the lock status
     * @return the new lock status, true if locked, false otherwise
     */
    public boolean toggleLock(){
        if(locked) unlock();
        else lock();
        return locked;
    }

    /**
     * Command to turn the camera on, taking the lock status is account
     */
    @Override
    public void turnOn() { if(!locked) super.turnOn(); }

    /**
     * Command to toggle the camera on or off, taking the lock status is account
     */
    @Override
    public boolean toggle() {
        if(!locked && !isOn()) super.turnOn();
        else turnOff();
        return isOn();
    }
}
