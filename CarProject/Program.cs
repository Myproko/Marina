// See https://aka.ms/new-console-template for more information

public class Car
{
    int Weight;
    float Price;
    public Car(int W, float Pr)
    {
        Weight = W;
        Price = Pr;
    }
    string CountryOfOrigin;
    string Brand;
    public void Sell()
    {
        bool selable = false;
        if (Price < 20000)
        {
            selable = true;
            Console.WriteLine("the car is sellable");
        }
        else
        {
            Console.WriteLine("the car is not sellable");
        }
    }
}
class Program
{
    static void Main(string[] args)
    {
        Car Honda = new Car(400, 5);
        Honda.Sell();
    }
}




