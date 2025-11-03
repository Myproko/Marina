using System;
using System.Diagnostics.Eventing.Reader;
using System.Drawing;
using System.Drawing.Text;
using System.IO;
using System.Linq.Expressions;
using System.Net;
using System.Net.Http;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Xml.Schema;

namespace TestWinForms
{
    public partial class Form1 : Form

    {
        public static string Homepage;
        public Form1()
        {
            InitializeComponent();
        }
               public void Search_Click(object sender, EventArgs e)
        {
            SendRequest(SearchLine.Text);
        }
        public async Task SendRequest(string destination, bool AddToHistory = true)
        {
            URLpanel.Controls.Clear();
            HttpClient client = new HttpClient();
            HttpResponseMessage response = new HttpResponseMessage();
            try
            {
                string content = await client.GetStringAsync(destination);
                response = await client.GetAsync(destination);
                response.EnsureSuccessStatusCode();
                //showing the response:
                MessageBox.Show(response.ToString());
                if (AddToHistory == true)
                {
                    addHistory(destination);
                }
                if (response.IsSuccessStatusCode)
                {
                    //display content in the window:
                    windowForHTTP.Text = content;
                    //get title from html content:
                    this.Text = GetTitle(content);
                    for (int i = 0; i < 5; ++i)
                    {
                        //calling Get5urls method to get 5 urls from content:
                        Get5urls(ref content);
                    }
                }
            }
            //catch exceptions for unsuccessful status codes try urls from  https://savanttools.com/test-http-status-codes
            catch (Exception ex)
            {
                windowForHTTP.Text = ex.Message;

                switch (ex.Message)
                {
                    case var s when ex.Message.Contains("404"):
                        windowForHTTP.Text = "Error: 404 Not Found";
                        break;
                    case var r when ex.Message.Contains("400"):
                        windowForHTTP.Text = "Error: 400 Bad Request";
                        break;
                    case var q when ex.Message.Contains("403"):
                        windowForHTTP.Text = "Error: 403 Forbidden";
                        break;
                    default:
                        windowForHTTP.Text = ex.Message;
                        break;
                }
                return;

            }

        }
        private string GetTitle(string content)
        //method to get title from html content
        {
            int ind1 = content.IndexOf("<title");
            int ind2 = content.IndexOf("</title>");
            if (ind1 == -1 || ind2 == -1 || ind2 <= ind1 + 7)
            {
                return "No Title Found";
            }
            ind1 = ind1 + 7;
            int length = ind2 - ind1;
            string title = content.Substring(ind1, length);
            title = title.Substring(title.IndexOf(">") + 1);
            //MessageBox.Show(title);
            return title;
        }
        private void Get5urls(ref string content)
        //get 5 urls from html content  
        {
            try
            {
                int ind1 = content.IndexOf("https://");
                content = content.Substring(ind1);
                char[] urlend = { ' ', '"', ';', ')' };
                int length = content.IndexOfAny(urlend);
                string url = content.Substring(0, length);

                content = content.Substring(length);

                LinkLabel urlLink = new LinkLabel();
                urlLink.Parent = URLpanel;
                urlLink.Text = url;
                urlLink.AutoSize = true;
                urlLink.Click += new EventHandler((s, ev) =>
                {
                    SearchLine.Text = urlLink.Text;
                    SendRequest(urlLink.Text);
                });
            }
            catch (Exception ex)
            {
                //just do nothing if there are no more urls
            }

        }
        private void Form1_Load(object sender, EventArgs e)
        //method to load homepage from file and display bookmarks
        {
            StreamReader sr = new StreamReader(@"..\..\Resources\homepageURL.txt");
            Homepage = sr.ReadLine();
            SendRequest(Homepage);
            sr.Close();
            SearchLine.Text = Homepage;
            DisplayBookmarks();
            DisplayHistory();
        }
        private void Home_Click(object sender, EventArgs e)
        //method to re-write homepage file with new homepage url
        {
            Homepage = SearchLine.Text;
            StreamWriter sw = new StreamWriter(@"..\..\Resources\homepageURL.txt");
            //Write a line of text
            sw.WriteLine(Homepage);
            sw.Close();
            //Old path - not anymore ))File.WriteAllText(@"C:\Users\Marina\OneDrive\Desktop\homepageURL.txt", Homepage);
        }
        private void Favorite(object sender, EventArgs e)
        //method to add bookmark button to FlowLayoutPanel and write bookmark to file

        {
            if (SearchLine.Text != "")
            {
                MessageBox.Show("Hi - if you want to delete bookmark later, right click the bookmark title button");
                string url = SearchLine.Text;
                MessageBox.Show(this.Text);
                Button button = new Button();
                // Set button properties
                //gettitle for button text
                button.Text = this.Text;
                button.Tag = url;
                //Bookmarks =FlowLayoutPanel
                button.Parent = Bookmarks;
                button.Click += new EventHandler(button_Click);
                Bookmark bookmark = new Bookmark(this.Text, url);
                Bookmark.WriteToMemory();


                button.MouseUp += new MouseEventHandler((s, ev) =>
                {
                    if (ev.Button == MouseButtons.Right)
                    {
                        button_RightClick(s, e);
                    }
                });
            }
            else
            {
                MessageBox.Show("No URL");
            }
        }
        private void button_RightClick(object sender, EventArgs e)
        //method to delete bookmark on right click
        {
            Bookmark.DeleteFromMemory((sender as Button).Tag as string);
            Bookmarks.Controls.Remove(sender as Button);
        }
        private void button_Click(object sender, EventArgs e)
        //method to navigate to bookmark url on button click
        {
            Button b = sender as Button;
            string url = b.Tag as string;
            SearchLine.Text = url;

            SendRequest(url);
        }
        public void DisplayBookmarks()
        //method to read bookmarks stored  in file from previouse use and display them as buttons in FlowLayoutPanel
        {
            StreamReader sr = new StreamReader(@"..\..\Resources\bookmarksURLS.txt");
            string bookmarkline;
            while ((bookmarkline = sr.ReadLine()) != null)
            {
                string[] parameters = bookmarkline.Split(',');

                Bookmark bookmark = new Bookmark(parameters[0], parameters[1]);
                Button button = new Button();
                // Set button properties
                //gettitle for button text
                button.Text = bookmark.Name;
                button.Tag = bookmark.URL;
                button.Parent = Bookmarks;
                button.Click += new EventHandler(button_Click);
                //create event handler for right click after it checks if the right button was clicked
                button.MouseUp += new MouseEventHandler((s, ev) =>
                {
                    if (ev.Button == MouseButtons.Right)
                    {
                        button_RightClick(s, ev);
                    }
                }
                );
            }
            sr.Close();
        }
        public void addHistory(string url)
        {
            if ((History.HistoryList.Count != 0))
            {
                if (url != History.HistoryList[History.HistoryList.Count - 1].url)
                {
                    History history = new History(url);
                    LinkLabel urlLink = new LinkLabel();
                    urlLink.Parent = HistoryPanel;
                    urlLink.Text = url;
                    urlLink.AutoSize = true;
                    urlLink.Click += new EventHandler
                        ((s, ev) =>
                        {
                            SearchLine.Text = urlLink.Text;
                            SendRequest(urlLink.Text);
                        }
                        );
                }
            }
            else
            {
                History history = new History(url);
                LinkLabel urlLink = new LinkLabel();
                urlLink.Parent = HistoryPanel;
                urlLink.Text = url;
                urlLink.AutoSize = true;
                urlLink.Click += new EventHandler
                    ((s, ev) =>
                    {
                        SearchLine.Text = urlLink.Text;
                        SendRequest(urlLink.Text);
                    }
                    );

            }
            if (History.HistoryList.Count > 5)
            {
                History.HistoryList.RemoveAt(0);
                HistoryPanel.Controls.RemoveAt(0);
            }
            History.WriteToMemory();
        }
        public void DisplayHistory()
        {
            StreamReader sr = new StreamReader(@"..\..\Resources\history.txt");
            string historyline;
            while ((historyline = sr.ReadLine()) != null)
            {
                History history = new History(historyline);
                LinkLabel urlLink = new LinkLabel();
                urlLink.Parent = HistoryPanel;
                urlLink.Text = historyline;
                urlLink.AutoSize = true;
                urlLink.Click += new EventHandler
                    ((s, ev) =>
                    {
                        SearchLine.Text = urlLink.Text;
                        SendRequest(urlLink.Text);
                    }
                    );
            }
            sr.Close();
        }

        private void Backword_Click(object sender, EventArgs e)
        {
            SendRequest(History.HistoryList[Math.Max(0, History.HistoryList.Count - 2)].url, false);
            SearchLine.Text = History.HistoryList[Math.Max(0, History.HistoryList.Count - 2)].url;
        }

        private void Forward_Click(object sender, EventArgs e)
        {
            SendRequest(History.HistoryList[Math.Min(History.HistoryList.Count - 1, History.HistoryList.Count)].url, false);
            SearchLine.Text = History.HistoryList[Math.Min(History.HistoryList.Count - 1, History.HistoryList.Count)].url;
        }

        private void Refresh_Click(object sender, EventArgs e)
        {
            SendRequest(SearchLine.Text);
        }

       
    }
}
    


                
                
                    
 

/*
 exceptions handling 2 unsuccessfull tries:
if-else version:
if (response.StatusCode == HttpStatusCode.BadRequest)
{
 MessageBox.Show("Error: 400 Bad Request");
return;
 }
else
 if
 (response.StatusCode == HttpStatusCode.Forbidden)
 {
MessageBox.Show("Error: 403 Forbidden");
 return;

try-catch version

{
   switch
       (response.StatusCode)
   {
       case HttpStatusCode.NotFound:
           MessageBox.Show("Error: 404 Not Found");
           break;
       case HttpStatusCode.BadRequest:
           MessageBox.Show("Error: 400 Bad Request");
           break;
       case HttpStatusCode.Forbidden:
           MessageBox.Show("Error: 403 Forbidden");
           break;
       default:
           MessageBox.Show("Error: " );
           break;
*/