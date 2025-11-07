// See https://aka.ms/new-console-template for more information
//Console.WriteLine("Hello, World!");
//using System;
//using System.Xml;


class Program
{

    // Multiply Matrices function
    static int[,] multiplyMatrices(int[,] a, int[,] b)
    {
        int a_rows = a.GetLength(0); //counts rows
        int a_columns = a.GetLength(1); //counts columns
        int b_rows = b.GetLength(0);
        int b_columns = b.GetLength(1);

        if ((b_columns != a_rows) || (b_rows != a_columns))
        {
            throw new Exception("Can't multiply matrices!");
        }

        int[,] result = new int[a_rows, b_columns];

        int sum = 0;

        for (int i = 0; i < a_rows; i++)          // looping thru rows of A
        {
            for (int j = 0; j < b_columns; j++)   // looping thru columns of B
            {
                sum = 0;
                for (int c = 0; c < a_columns; c++)  // columns of A = rows of B
                {
                    sum += a[i, c] * b[c, j];
                }
                result[i, j] = sum;
            }
        }
        return result;
    }

    // Multiply matrices function done

    //Point of entry for the program
    static void Main(string[] args)
    {
        int[,] first = new int[,]
        {
        {1, 2, 3},
        {4, 5, 6}
        };

        int[,] second = new int[,]
        {
        {1,2},
        {3,4},
        {5,6}
        };

        int[,] result = multiplyMatrices(first, second); //calls out the Multiply Matrices function

        for (int i = 0; i < result.GetLength(0); i++)
        {
            for (int j = 0; j < result.GetLength(1); j++)
            {
                Console.Write(result[i, j] + " ");
            }
            Console.WriteLine(Environment.NewLine);
        }

    }
}
