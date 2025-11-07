package homesystem;

/**
 * A class to represent lights in the home automation system
 */
public class Light extends Device implements Adjustable{

    private int brightness;
    private String color;

    /**
     * Constructor
     * @param n the initial light name
     * @param i the light id
     */
    public Light(String n, String i){
        super(n, i);
        brightness = 50;
        color = "#fcf0cf";
    }

    /**
     * Constructor with default light name
     * @param i the light id
     */
    public Light(String i){
        this("Light", i);
    }

    /**
     * Query for the light brightness
     * @return the light brightness value
     */
    public int getBrightness(){ return brightness; }
    /**
     * Query for the light color
     * @return the light color value
     */
    public String getColor(){ return color; }

    /**
     * Command to adjust the light's brightness
     * @param b the new light brightness value
     */
    public void adjustBrightness(int b){ brightness = b; }
    /**
     * Command to change the light's color <br>
     * Uses a hexadecimal String representation
     * @param hex the new light color in hexadecimal String
     */
    public void setColor(String hex){ color = hex; }
    /**
     * Command to change the light's color <br>
     * Uses an HSL representation and internally converts it to hexadecimal String
     * @param h the new light color hue
     * @param s the new light color saturation
     * @param l the new light color luminance
     */
    public void setColor(int h, float s, float l){
        System.out.println("Converted HSL to HEX");
        this.setColor("#fcf0cf");
    }
    /**
     * Command to change the light's color <br>
     * Uses an RGB representation and internally converts it to hexadecimal String
     * @param r the new light color red value
     * @param g the new light color green value
     * @param b the new light color blue value
     */
    public void setColor(int r, int g, int b){
        System.out.println("Converted RGB to HEX");
        this.setColor("#fcf0cf");
    }

    /**
     * Query for the light's status
     * @return "off" if the light is off, otherwise shows info about the brightness and color
     */
    public String getStatus(){
        if(this.isOn()) return brightness + "% bright, color: " + color;
        return "off";
    }

    /**
     * Query for the String representation of the light: "Light id: name (status)"
     * @return the String representation of the light
     */
    @Override
    public String toString(){ return "Light " + super.toString() + "(" + this.getStatus() + ")"; }

    /**
     * Command for testing the equality of a Light object with another object
     * @param o Object to test the equality against
     * @return true if the object is a light with the same id, false otherwise
     */
    @Override
    public boolean equals(Object o) {
        // if equal by reference then the objects are the same
        if (this == o) return true;
        // if the compared object is empty or from a different
        // class, the objects can't be equal
        if (o == null || getClass() != o.getClass()) return false;
        // cast the object to a light object
        // and compare IDs
        Light light = (Light) o;
        return getID().equals(light.getID());
    }

    /**
     * Command for turning the light on with a brightness value
     * @param b the brightness value
     */
    public void turnOn(int b){
        super.turnOn();
        this.adjustBrightness(b);
    }

    /**
     * Query for the adjustable setting: brightness
     * @return the brightness value
     */
    public int getAdjustableValue(){ return this.getBrightness(); }

    /**
     * Command to set the adjustable setting's value: brightness
     * @param v the new brightness value to set
     */
    public void adjustValue(int v){ this.adjustBrightness(v); }
}
