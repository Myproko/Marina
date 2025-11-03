package homesystem;

import java.util.ArrayList;

/**
 * A class to represent rooms in the home automation system
 */
public class Room {

    private String name;
    private ArrayList<Device> devices;

    /**
     * Constructor
     * @param n the room's name
     */
    public Room(String n){
        name = n;
        devices = new ArrayList<>();
    }

    /**
     * Command to add a device to list of devices in the room
     * @param d the device to add
     */
    public void addDevice(Device d){
        devices.add(d);
        d.addToRoom(this);
    }

    /**
     * Command to get the list of devices in the room as a String with details
     * @return the list of devices, as a String
     */
    public String listDevices(){
        StringBuilder list = new StringBuilder("Devices in "+name+":\n");
        for(Device d: devices){
            list.append(d.toString()).append("\n");
        }
        return list.toString();
    }
}
