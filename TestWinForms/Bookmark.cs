using System.Collections.Generic;
using System.IO;
namespace TestWinForms
{
    public partial class Bookmark : Form1
        

    //create a class to store bookmarks with static methods to add, remove and get bookmarks in file
    {
        public string Name { get; set; }
        public string URL { get; set; }
        public static List<Bookmark> bookmarks = new List<Bookmark>();
        public Bookmark(string name, string url)
        {
            this.Name = name;
            this.URL = url;
            bookmarks.Add(this);
        }
        public static void WriteToMemory()
        {

            StreamWriter sw = new StreamWriter(@"..\..\Resources\bookmarksURLS.txt");
            foreach (Bookmark b in bookmarks)
            {
                sw.WriteLine(b.Name + "," + b.URL);
            }
            //Write a line of text

            sw.Close();
            //File.WriteAllText(@"C:\Users\Marina\OneDrive\Desktop\homepageURL.txt", Homepage);
        }
        public static void DeleteFromMemory(string url)
        {
            foreach (Bookmark b in bookmarks)
            {
                string   link = b.URL;
                if (link == url)
                {
                    bookmarks.Remove(b);
                    break;
                }
                 }
            WriteToMemory();
            //Write a line of text
        }
        
        }
        }

    



