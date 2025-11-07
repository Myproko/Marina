namespace Exercise0
{
    using System;
    class Enums
{
    // this defines a new type, with name Weekdays, and lists possible values
    enum Weekdays { Mon, Tue, Wed, Thu, Fri, Sat, Sun };
    // this defines another type, to distinguish weekdays; we will use it later
    enum Days { WeekDay, WeekEnd };

    // a standard main function, that takes arguments from the command-line 
    static void Main(string[] args)
    {
        //System.Console.WriteLine("cmdline arg: {0}", args[0]);
        if (args.Length != 1)
        { // expect 1 arg: day of week
            System.Console.WriteLine("Usage: exercise0 <weekday>");
        }
        else
        {
            Weekdays z = Weekdays.Mon; // needed for parsing
            Weekdays d = (Weekdays)Enum.Parse(z.GetType(), args[0]);
            Enums.Test(d); // call the test method in this class
                           // Enums.Test(Weekdays.Sun); // test with a fixed input
        }
    }
    /// <summary>
    /// This method runs series of tests on the methods in this class.
    /// </summary>
    /// <seealso cref="NextDay(Weekdays)"> This is the main method that is tested. </seealso>
    static void Test(Weekdays d)
    {
        Days d0 = Enums.WhatDay(d);
        System.Console.WriteLine("Testing some enums now...");
        Weekdays e = Enums.NextDay(d);
        Days e0 = Enums.WhatDay(e);
        System.Console.WriteLine("Tomorrow of " + d + " (a " + d0 + ") is " + e + " (a " + e0 + ")");
        Weekdays f = Enums.NextDay1(d);
        Days f0 = Enums.WhatDay(f);
        System.Console.WriteLine("Tomorrow of " + d + " (a " + d0 + ") is " + f + " (a " + f0 + ")");
    }

    // We want to define a method that takes a day and returns the next day
    /// <summary>
    /// Take a day as input and return the next day as a result.
    /// </summary>
    /// <param name="d"> The input day  </param>
    /// <returns>Returns the next day of the week. </returns>
    static Weekdays NextDay(Weekdays d)
    {
        // using a switch statement here (easiest solution)
        Weekdays e;  // declare a variable which will hold the result
        switch (d)
        { // examine the value in d, and check if it's one of the values below
            case Weekdays.Mon: e = Weekdays.Tue; break; // if the input was Mon we return Tue, etc
            case Weekdays.Tue: e = Weekdays.Wed; break;
            case Weekdays.Wed: e = Weekdays.Thu; break;
            case Weekdays.Thu: e = Weekdays.Fri; break;
            case Weekdays.Fri: e = Weekdays.Sat; break;
            case Weekdays.Sat: e = Weekdays.Sun; break;
            case Weekdays.Sun: e = Weekdays.Mon; break;
            default: e = Weekdays.Mon; break; // needs default case to init e
        }
        return e; // return the value we have assigned to
    }

    /// <summary>
    /// Take a day as input and return the next day as a result.
    /// An alternative, shorter implementation.
    /// </summary>
    /// <param name="d"> The input day  </param>
    /// <returns>Returns the next day of the week. </returns>
    static Weekdays NextDay1(Weekdays d)
    {
        Weekdays e = (Weekdays)((int)(d + 1) % 7);     // uses implicit conversions and int as basetype
        return e;
    }

    /// <summary>
    /// Take a day as input and return whether it is a WeekDay or a WeekEnd.
    /// </summary>
    /// <param name="d"> The input day  </param>
    /// <returns>Returns WeekDay or WeekEnd. </returns>
    static Days WhatDay(Weekdays d)
    {
        switch (d)
        {
            case Weekdays.Mon:
            case Weekdays.Tue:
            case Weekdays.Wed:
            case Weekdays.Thu:
            case Weekdays.Fri:
                return Days.WeekDay;
            case Weekdays.Sat:
            case Weekdays.Sun:
                return Days.WeekEnd;
            default:
                return Days.WeekEnd;
        }
    }
} /* class */
} /* namespace */
