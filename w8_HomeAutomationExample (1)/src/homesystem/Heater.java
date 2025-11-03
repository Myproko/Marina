package homesystem;

/**
 * A class to represent heaters in the home automation system
 */
public class Heater extends Device implements Adjustable{

    private int temperature;
    private String mode;

    /**
     * Constructor
     * @param n the initial heater name
     * @param i the heater id
     */
    public Heater(String n, String i){
        super(n, i);
        temperature = 50;
        mode = "#fcf0cf";
    }

    /**
     * Constructor with default heater name
     * @param i the speaker id
     */
    public Heater(String i){
        this("Heater", i);
    }

    /**
     * Query for the heater temperature
     * @return the heater temperature value
     */
    public int getTemperature(){ return temperature; }
    /**
     * Query for the heater mode
     * @return the heater mode value
     */
    public String getMode(){ return mode; }

    /**
     * Command to adjust the heater's temperature
     * @param t the new heater temperature value
     */
    public void adjustTemperature(int t){ temperature = t; }
    /**
     * Command to change the heater's mode
     * @param m the new heater mode
     */
    public void setMode(String m){ mode = m; }

    /**
     * Query for the heater's status
     * @return "off" if the heater is off, otherwise shows info about the temperature and mode
     */
    public String getStatus(){
        if(this.isOn()) return "temperature: " + temperature + "°C, " + mode + " mode";
        return "off";
    }

    /**
     * Query for the String representation of the heater: "Heater id: name (status)"
     * @return the String representation of the heater
     */
    @Override
    public String toString(){ return "Heater " + super.toString() + "(" + this.getStatus() + ")"; }

    /**
     * Query for the adjustable setting: temperature
     * @return the temperature value
     */
    public int getAdjustableValue(){ return this.getTemperature(); }

    /**
     * Command to set the adjustable setting's value: temperature
     * @param v the new temperature value to set
     */
    public void adjustValue(int v){ this.adjustTemperature(v); }
}
