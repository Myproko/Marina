package homesystem;

/**
 * An interface for devices that have an adjustable setting.
 */
public interface Adjustable {

    /**
     * Queries the adjustable setting's value
     * @return the value of the adjustable setting
     */
    int getAdjustableValue();

    /**
     * Sets the adjustable setting's value
     * @param v the new value to set
     */
    void adjustValue(int v);
}
