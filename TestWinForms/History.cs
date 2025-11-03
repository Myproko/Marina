using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestWinForms
{
    public class History
    {
        public static List<History> HistoryList = new List<History>();
        public string url;
        public History(string URL)
        {
         this.url = URL;
         HistoryList.Add(this);
        }
        public static void WriteToMemory()
        {
            StreamWriter sw = new StreamWriter(@"..\..\Resources\history.txt");
            foreach (History h in HistoryList)
            {
                sw.WriteLine(h.url);
            }
            sw.Close();
        }
        

    }
}
