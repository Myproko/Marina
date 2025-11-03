using System.Drawing;

namespace TestWinForms
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.SearchLine = new System.Windows.Forms.RichTextBox();
            this.Search = new System.Windows.Forms.Button();
            this.windowForHTTP = new System.Windows.Forms.RichTextBox();
            this.Star = new System.Windows.Forms.Button();
            this.Home = new System.Windows.Forms.Button();
            this.Bookmarks = new System.Windows.Forms.FlowLayoutPanel();
            this.URLpanel = new System.Windows.Forms.FlowLayoutPanel();
            this.HistoryPanel = new System.Windows.Forms.FlowLayoutPanel();
            this.Forward = new System.Windows.Forms.Button();
            this.Backword = new System.Windows.Forms.Button();
            this.Refresh = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // SearchLine
            // 
            this.SearchLine.AutoWordSelection = true;
            this.SearchLine.BackColor = System.Drawing.SystemColors.Window;
            this.SearchLine.Location = new System.Drawing.Point(68, 28);
            this.SearchLine.Multiline = false;
            this.SearchLine.Name = "SearchLine";
            this.SearchLine.ScrollBars = System.Windows.Forms.RichTextBoxScrollBars.None;
            this.SearchLine.Size = new System.Drawing.Size(822, 46);
            this.SearchLine.TabIndex = 0;
            this.SearchLine.Text = "";
            // 
            // Search
            // 
            this.Search.BackColor = System.Drawing.Color.LavenderBlush;
            this.Search.Font = new System.Drawing.Font("Times New Roman", 10F, ((System.Drawing.FontStyle)((System.Drawing.FontStyle.Bold | System.Drawing.FontStyle.Italic))), System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.Search.Location = new System.Drawing.Point(796, 28);
            this.Search.Name = "Search";
            this.Search.Size = new System.Drawing.Size(152, 50);
            this.Search.TabIndex = 1;
            this.Search.Text = "Search";
            this.Search.UseVisualStyleBackColor = false;
            this.Search.Click += new System.EventHandler(this.Search_Click);
            // 
            // windowForHTTP
            // 
            this.windowForHTTP.BackColor = System.Drawing.SystemColors.ControlLightLight;
            this.windowForHTTP.Location = new System.Drawing.Point(68, 165);
            this.windowForHTTP.Name = "windowForHTTP";
            this.windowForHTTP.ReadOnly = true;
            this.windowForHTTP.Size = new System.Drawing.Size(880, 300);
            this.windowForHTTP.TabIndex = 2;
            this.windowForHTTP.Text = "";
            // 
            // Star
            // 
            this.Star.BackColor = System.Drawing.Color.LavenderBlush;
            this.Star.Image = global::TestWinForms.Properties.Resources.pic;
            this.Star.Location = new System.Drawing.Point(68, 80);
            this.Star.Name = "Star";
            this.Star.Size = new System.Drawing.Size(77, 52);
            this.Star.TabIndex = 5;
            this.Star.UseVisualStyleBackColor = false;
            this.Star.Click += new System.EventHandler(this.Favorite);
            // 
            // Home
            // 
            this.Home.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(255)))), ((int)(((byte)(192)))));
            this.Home.Image = global::TestWinForms.Properties.Resources.etot;
            this.Home.Location = new System.Drawing.Point(1209, 31);
            this.Home.Name = "Home";
            this.Home.Size = new System.Drawing.Size(65, 49);
            this.Home.TabIndex = 3;
            this.Home.UseVisualStyleBackColor = false;
            this.Home.Click += new System.EventHandler(this.Home_Click);
            // 
            // Bookmarks
            // 
            this.Bookmarks.AutoSize = true;
            this.Bookmarks.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.Bookmarks.BackColor = System.Drawing.SystemColors.ControlLightLight;
            this.Bookmarks.Location = new System.Drawing.Point(184, 80);
            this.Bookmarks.Name = "Bookmarks";
            this.Bookmarks.Size = new System.Drawing.Size(0, 0);
            this.Bookmarks.TabIndex = 7;
            this.Bookmarks.Tag = "Bookmarks";
            // 
            // URLpanel
            // 
            this.URLpanel.BackColor = System.Drawing.SystemColors.ControlLightLight;
            this.URLpanel.FlowDirection = System.Windows.Forms.FlowDirection.TopDown;
            this.URLpanel.Location = new System.Drawing.Point(68, 489);
            this.URLpanel.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.URLpanel.Name = "URLpanel";
            this.URLpanel.Padding = new System.Windows.Forms.Padding(0, 0, 0, 31);
            this.URLpanel.Size = new System.Drawing.Size(880, 118);
            this.URLpanel.TabIndex = 9;
            this.URLpanel.Tag = "5 clikable Links ( URLs)  from the page above";
            // 
            // HistoryPanel
            // 
            this.HistoryPanel.BackColor = System.Drawing.SystemColors.ControlLightLight;
            this.HistoryPanel.Location = new System.Drawing.Point(1004, 233);
            this.HistoryPanel.Name = "HistoryPanel";
            this.HistoryPanel.Size = new System.Drawing.Size(270, 232);
            this.HistoryPanel.TabIndex = 10;
            // 
            // Forward
            // 
            this.Forward.BackColor = System.Drawing.Color.LavenderBlush;
            this.Forward.Font = new System.Drawing.Font("Russo One", 9.999999F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.Forward.Location = new System.Drawing.Point(1209, 164);
            this.Forward.Name = "Forward";
            this.Forward.Size = new System.Drawing.Size(65, 47);
            this.Forward.TabIndex = 11;
            this.Forward.Text = "-->";
            this.Forward.UseVisualStyleBackColor = false;
            this.Forward.Click += new System.EventHandler(this.Forward_Click);
            // 
            // Backword
            // 
            this.Backword.BackColor = System.Drawing.Color.LavenderBlush;
            this.Backword.Font = new System.Drawing.Font("Russo One", 9.999999F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.Backword.Location = new System.Drawing.Point(1004, 165);
            this.Backword.Name = "Backword";
            this.Backword.Size = new System.Drawing.Size(59, 47);
            this.Backword.TabIndex = 12;
            this.Backword.Text = "<----";
            this.Backword.UseVisualStyleBackColor = false;
            this.Backword.Click += new System.EventHandler(this.Backword_Click);
            // 
            // Refresh
            // 
            this.Refresh.BackColor = System.Drawing.Color.LavenderBlush;
            this.Refresh.Font = new System.Drawing.Font("Times New Roman", 10F, ((System.Drawing.FontStyle)((System.Drawing.FontStyle.Bold | System.Drawing.FontStyle.Italic))), System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.Refresh.Location = new System.Drawing.Point(1087, 31);
            this.Refresh.Name = "Refresh";
            this.Refresh.Size = new System.Drawing.Size(102, 49);
            this.Refresh.TabIndex = 13;
            this.Refresh.Text = "Refresh";
            this.Refresh.UseVisualStyleBackColor = false;
            this.Refresh.Click += new System.EventHandler(this.Refresh_Click);
            // 
            // label1
            // 
            this.label1.BackColor = System.Drawing.SystemColors.ControlLightLight;
            this.label1.Font = new System.Drawing.Font("Times New Roman", 10F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.Location = new System.Drawing.Point(73, 433);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(103, 32);
            this.label1.TabIndex = 14;
            this.label1.Text = "HTML";
            // 
            // label2
            // 
            this.label2.BackColor = System.Drawing.Color.LavenderBlush;
            this.label2.Font = new System.Drawing.Font("Times New Roman", 10F, ((System.Drawing.FontStyle)((System.Drawing.FontStyle.Bold | System.Drawing.FontStyle.Italic))), System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.Location = new System.Drawing.Point(1083, 164);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(106, 47);
            this.label2.TabIndex = 15;
            this.label2.Text = "History";
            this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.AutoSize = true;
            this.BackColor = System.Drawing.SystemColors.GradientActiveCaption;
            this.BackgroundImage = global::TestWinForms.Properties.Resources.photo;
            this.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.ClientSize = new System.Drawing.Size(1322, 708);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.Refresh);
            this.Controls.Add(this.Backword);
            this.Controls.Add(this.Forward);
            this.Controls.Add(this.HistoryPanel);
            this.Controls.Add(this.URLpanel);
            this.Controls.Add(this.Bookmarks);
            this.Controls.Add(this.Star);
            this.Controls.Add(this.Home);
            this.Controls.Add(this.Search);
            this.Controls.Add(this.SearchLine);
            this.Controls.Add(this.windowForHTTP);
            this.Name = "Form1";
            this.Text = "Form1.text";
            this.Load += new System.EventHandler(this.Form1_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private System.Windows.Forms.Button Search;
        private System.Windows.Forms.RichTextBox windowForHTTP;
        public System.Windows.Forms.RichTextBox SearchLine;
        private System.Windows.Forms.Button Home;
        private System.Windows.Forms.Button Star;
        private System.Windows.Forms.FlowLayoutPanel Bookmarks;
        private System.Windows.Forms.FlowLayoutPanel URLpanel;
        private System.Windows.Forms.FlowLayoutPanel HistoryPanel;
        private System.Windows.Forms.Button Forward;
        private System.Windows.Forms.Button Backword;
        private System.Windows.Forms.Button Refresh;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
    }
}

