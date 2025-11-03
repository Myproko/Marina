package homesystem;

/**
 * A class to represent speakers in the home automation system
 */
public class Speaker extends Device implements Adjustable, Lockable{

    private int volume;
    private String source;
    private boolean locked;

    /**
     * Constructor
     * @param n the initial speaker name
     * @param i the speaker id
     */
    public Speaker(String n, String i){
        super(n, i);
        volume = 50;
        source = "";
        locked = true;
    }

    /**
     * Constructor with default speaker name
     * @param i the speaker id
     */
    public Speaker(String i){
        this("Speaker", i);
    }

    /**
     * Query for the speaker volume
     * @return the speaker volume value
     */
    public int getVolume(){ return volume; }
    /**
     * Query for the speaker source
     * @return the speaker source value
     */
    public String getSource(){ return source; }
    /**
     * Query to chceck if the speaker has a source
     * @return true if there is a source, false otherwise
     */
    public boolean hasSource(){ return !source.isEmpty(); }

    /**
     * Command to adjust the speaker's volume
     * @param v the new speaker volume value
     */
    public void adjustVolume(int v){ volume = v; }
    /**
     * Command to change the speaker's source
     * @param s the new speaker source
     */
    public void setSource(String s){ source = s; }

    /**
     * Query for the speaker's status
     * @return "off" if the speaker is off, otherwise shows info about the volume and source
     */
    public String getStatus(){
        String lock = isLocked() ? "locked" : "unlocked";
        if(this.isOn()) {
            String src = this.hasSource() ? "source: " + source : "no source";
            return "volume: " + volume + "&, " + src + ", " + lock;
        }
        return "off, " + lock;
    }

    /**
     * Query for the String representation of the speaker: "Speaker id: name (status)"
     * @return the String representation of the speaker
     */
    @Override
    public String toString(){ return "Speaker " + super.toString() + "(" + this.getStatus() + ")"; }

    /**
     * Query for the adjustable setting: volume
     * @return the volume value
     */
    public int getAdjustableValue(){ return this.getVolume(); }

    /**
     * Command to set the adjustable setting's value: volume
     * @param v the new volume value to set
     */
    public void adjustValue(int v){ this.adjustVolume(v); }

    /**
     * Command to lock the speaker <br> Also turns it off
     */
    public void lock(){ locked = true; turnOff(); }

    /**
     * Command to unlock the speaker
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
     * Command to turn the speaker on, taking the lock status is account
     */
    @Override
    public void turnOn() { if(!locked) super.turnOn(); }

    /**
     * Command to toggle the speaker on or off, taking the lock status is account
     */
    @Override
    public boolean toggle() {
        if(!locked && !isOn()) super.turnOn();
        else turnOff();
        return isOn();
    }
}
