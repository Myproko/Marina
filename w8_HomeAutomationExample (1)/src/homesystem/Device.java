package homesystem;

/**
 * An abstract class to represent devices in the home automation system
 */
public abstract class Device {

    private String name;
    private String id;
    private boolean on;
    private Room room;

    /**
     * Constructor
     * @param n the initial device name
     * @param i the device id
     */
    protected Device(String n, String i){
        name = n; id = i; on = false; room = null;
    }

    /**
     * Query for the device name
     * @return the device's name
     */
    public String getName(){ return name; }
    /**
     * Query for the device id
     * @return the device's id
     */
    public String getID(){ return id; }
    /**
     * Query for the device on/off status
     * @return true if the device is on, false otherwise
     */
    public boolean isOn(){ return on; }
    /**
     * Query for the device room
     * @return the device's room
     */
    public Room getRoom(){ return room; }

    /**
     * Command to update the device's name
     * @param n the new name to set
     */
    public void changeName(String n){ name = n; }
    /**
     * Command to turn on the device
     */
    public void turnOn(){ on = true; }
    /**
     * Command to turn off the device
     */
    public void turnOff(){ on = false; }
    /**
     * Command to toggle the device's on/off status <br>
     * Change to on if currently off and vice-versa
     * @return the new on/off status of the device
     */
    public boolean toggle(){ if(on) turnOff(); else turnOn(); return on; }
    /**
     * Command to register a new room for the device
     * @param r the new room for the device
     */
    public void addToRoom(Room r){ room = r; }
    /**
     * Query for whether the device is registered to a room or not
     * @return true if the device is registered to a room, false otherwise
     */
    public boolean inRoom() { return room != null; }

    /**
     * Query for the String representation of the device: "id: name"
     * @return the String representation of the device
     */
    @Override
    public String toString() { return id + ": " + name; }

    /**
     * Queries the status of the device <br>
     * Abstract has each subclass can provide different statuses
     * @return the device's status
     */
    public abstract String getStatus();

}
