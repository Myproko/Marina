package homesystem;

import java.util.ArrayList;

/**
 * A class to represent homes in the home automation system
 */
public class Home {

    private String name;
    private ArrayList<Device> devices;
    private ArrayList<Room> rooms;

    /**
     * Constructor
     * @param name the home's name
     */
    public Home(String name){
        this.name = name;
        devices = new ArrayList<>();
        rooms = new ArrayList<>();
    }

    /**
     * Command to create a new room in the home
     * @param name the room name
     * @return the room object
     */
    public Room createRoom(String name){
        Room room = new Room(name);
        rooms.add(room);
        return room;
    }

    /**
     * Command to create a new light in the home
     * @param name the light name
     * @param id the light id
     * @return the light object
     */
    public Light createLight(String name, String id){
        Light light = new Light(name, id);
        devices.add(light);
        return light;
    }

    /**
     * Command to create a new speaker in the home
     * @param name the speaker name
     * @param id the speaker id
     * @return the speaker object
     */
    public Speaker createSpeaker(String name, String id){
        Speaker speaker = new Speaker(name, id);
        devices.add(speaker);
        return speaker;
    }

    /**
     * Command to create a new heater in the home
     * @param name the heater name
     * @param id the heater id
     * @return the heater object
     */
    public Heater createHeater(String name, String id){
        Heater heater = new Heater(name, id);
        devices.add(heater);
        return heater;
    }

    /**
     * Command to create a new camera in the home
     * @param name the camera name
     * @param id the camera id
     * @return the camera object
     */
    public Camera createCamera(String name, String id){
        Camera camera = new Camera(name, id);
        devices.add(camera);
        return camera;
    }

    /**
     * Query for the list of all lockable devices in the home
     * @return the list of lockable devices in the home
     */
    public ArrayList<Lockable> listLockables(){
        ArrayList<Lockable> list = new ArrayList<>();
        for(Device d: devices){
            if(d instanceof Lockable) list.add((Lockable) d);
        }
        return list;
    }

    /**
     * Query for the list of all adjustable devices in the home
     * @return the list of adjustable devices in the home
     */
    public ArrayList<Adjustable> listAdjustables(){
        ArrayList<Adjustable> list = new ArrayList<>();
        for(Device d: devices){
            if(d instanceof Adjustable) list.add((Adjustable) d);
        }
        return list;
    }

    /**
     * Query for the list of all devices in the home
     * @return the list of devices in the home
     */
    public ArrayList<Device> listAllDevices(){
        return devices;
    }

    /**
     * Query for the list of all rooms in the home
     * @return the list of rooms in the home
     */
    public ArrayList<Room> listAllRooms(){
        return rooms;
    }

    /**
     * Method to generate a sample home system
     * @return a home object populated with rooms and devices
     */
    public static Home generateHome(){
        Home h = new Home("My Home");
        Room office = h.createRoom("Office");
        Room kitchen = h.createRoom("Kitchen");
        Room living = h.createRoom("Living Room");
        Room hall = h.createRoom("Hallway");

        Light l1 = h.createLight("Floor lamp", "l79");
        Light l2 = h.createLight("Living room", "l52");
        Light l3 = h.createLight("Desk", "l83");
        Light l4 = h.createLight("Counter top", "l21");
        Speaker s1 = h.createSpeaker("Office speaker", "s67");
        Speaker s2 = h.createSpeaker("Kitchen speaker", "s50");
        Camera c1 = h.createCamera("Entrance camera", "c45");

        office.addDevice(l3);office.addDevice(s1);
        living.addDevice(l1);living.addDevice(l2);
        kitchen.addDevice(l4);kitchen.addDevice(s2);
        hall.addDevice(c1);

        return h;
    }
}
