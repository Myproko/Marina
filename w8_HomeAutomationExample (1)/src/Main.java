import homesystem.Home;
import homesystem.Room;

public class Main {
    public static void main(String[] args) {
        Home home = Home.generateHome();

        System.out.println("\nAdjustables:");
        System.out.println(home.listAdjustables());
        System.out.println("\nLockables:");
        System.out.println(home.listLockables());
        System.out.println("\nDevices:");
        System.out.println(home.listAllDevices());

        for(Room r: home.listAllRooms()){
            System.out.println(r.listDevices());
        }


    }
}
