using LiveCharts;
using LiveCharts.Definitions.Charts;
using LiveCharts.Wpf;
using NAudio.Wave;
using Newtonsoft.Json;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Media;
using System.Net.Http;
using System.Reflection.Emit;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;
using System.Windows.Shapes;

namespace FoodTracking
{
    public partial class MainWindow : Window
    {
        public ObservableCollection<DataItem> DataItems { get; set; }

        private HttpClient _httpClient;
        private bool _isListening;

        // TextBlock referanslarını saklamak için bir liste
        private List<TextBlock> PieTitles { get; set; }

        public MainWindow()
        {
            InitializeComponent();

            _httpClient = new HttpClient();
            _isListening = true;
            StartSseListener();

            // SeriesCollection for each PieChart
            PieSeries1 = CreatePieSeries();
            PieSeries2 = CreatePieSeries();
            PieSeries3 = CreatePieSeries();
            PieSeries4 = CreatePieSeries();
            PieSeries5 = CreatePieSeries();
            PieSeries6 = CreatePieSeries();
            PieSeries7 = CreatePieSeries();
            PieSeries8 = CreatePieSeries();

            PieTitles = new List<TextBlock>
            {
                PieTitle1, PieTitle2, PieTitle3, PieTitle4, PieTitle5, PieTitle6, PieTitle7, PieTitle8
            };

            DataItems = new ObservableCollection<DataItem>();

            // DataGrid'e veri bağlayın
            DataGridTable.ItemsSource = DataItems;
            AddDummyData();
            DataContext = this;
        }
        private void AddDummyData()
        {
            // Örnek veri ekleyin
           // DataItems.Add(new DataItem { Information = "Rice supply has reached a critical threshold; a refill is recommended.", Date = "08.01.2025 - 14:57" });
          //  DataItems.Add(new DataItem { Information = "Rice is running out faster than expected.", Date = "08.01.2025 - 14:55" });

        }


        private async void StartSseListener()
        {
            try
            {
                using var request = new HttpRequestMessage(HttpMethod.Get, "http://127.0.0.1:5000/stream");
                using var response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);

                response.EnsureSuccessStatusCode();

                using var stream = await response.Content.ReadAsStreamAsync();
                using var reader = new StreamReader(stream, Encoding.UTF8);


                while (_isListening)
                {
                    var line = await reader.ReadLineAsync();
                    if (line != null && line.StartsWith("data:"))
                    {
                        string jsonData = line.Substring(5).Trim();

                        // Parse JSON data
                        var foodList = string.IsNullOrEmpty(jsonData)
                            ? new List<Food>() // Eğer jsonData boşsa, boş bir liste oluştur
                            : JsonConvert.DeserializeObject<List<Food>>(jsonData) ?? new List<Food>(); // Deserialize et, null ise yine boş liste oluştur


                        if (foodList != null)
                        {
                            Dispatcher.Invoke(() =>
                            {
                                UpdatePieSeries(foodList);
                            });
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"SSE bağlantı hatası: {ex.Message}");
            }
        }

        int count = 0;

        // PieSeries'leri günceller
        private void UpdatePieSeries(List<Food> foodList)
        {
            // PieSeries listesi
            var pieSeriesList = new[]
            {
                new { Series = PieSeries1, Chart = PieChart1, Title = PieTitle1, CriticalIcon = CriticalIcon1, WarningIcon = WarningIcon1, Rectangle = Rectangle1 },
                new { Series = PieSeries2, Chart = PieChart2, Title = PieTitle2, CriticalIcon = CriticalIcon2, WarningIcon = WarningIcon2, Rectangle = Rectangle2 },
                new { Series = PieSeries3, Chart = PieChart3, Title = PieTitle3, CriticalIcon = CriticalIcon3, WarningIcon = WarningIcon3, Rectangle = Rectangle3 },
                new { Series = PieSeries4, Chart = PieChart4, Title = PieTitle4, CriticalIcon = CriticalIcon4, WarningIcon = WarningIcon4, Rectangle = Rectangle4 },
                new { Series = PieSeries5, Chart = PieChart5, Title = PieTitle5, CriticalIcon = CriticalIcon5, WarningIcon = WarningIcon5, Rectangle = Rectangle5 },
                new { Series = PieSeries6, Chart = PieChart6, Title = PieTitle6, CriticalIcon = CriticalIcon6, WarningIcon = WarningIcon6, Rectangle = Rectangle6 },
                new { Series = PieSeries7, Chart = PieChart7, Title = PieTitle7, CriticalIcon = CriticalIcon7, WarningIcon = WarningIcon7, Rectangle = Rectangle7 },
                new { Series = PieSeries8, Chart = PieChart8, Title = PieTitle8, CriticalIcon = CriticalIcon8, WarningIcon = WarningIcon8, Rectangle = Rectangle8 }
            };

            foreach (var food in foodList)
            {
                bool isUpdated = false;

                for(int i = 0; i < 8; i++)
                {
                    var emptyPie = pieSeriesList[i].Series[0] as PieSeries;
                    var fillPie = pieSeriesList[i].Series[1] as PieSeries;

                    if (fillPie != null && pieSeriesList[i].Title.Text == food.Name)
                    {

                        pieSeriesList[i].Chart.Visibility = Visibility.Visible;
                        // FillRate yüzdelik olarak ayarlanır
                        double fillRate = Math.Round(food.FillRate, 2); // Yüzdelik değeri 2 ondalık basamakla yuvarla
                        double emptyRate = Math.Round(100 - fillRate, 2);

                        Image cricitalIcon = pieSeriesList[i].CriticalIcon;
                        Image warningIcon = pieSeriesList[i].WarningIcon;
                        Rectangle rect = pieSeriesList[i].Rectangle;

                        UpdateIconsAndRectangleVisibility(fillRate, cricitalIcon, warningIcon, rect);

                        // Güncelle
                        fillPie.Values[0] = fillRate;
                        emptyPie.Values[0] = emptyRate;
                        UpdatePieSeriesColor(fillPie, fillRate);
                        //pie.Values[1] = 100 - food.FillRate;
                        isUpdated = true;
                        break;
                    }
                }

                if (!isUpdated)
                {
                    for(int i = 0; i < 8; i++)
                    {
                        var emptyPie = pieSeriesList[i].Series[0] as PieSeries;
                        var fillPie = pieSeriesList[i].Series[1] as PieSeries;
                        if (fillPie != null && string.IsNullOrEmpty(pieSeriesList[i].Title.Text))
                        {
                            pieSeriesList[i].Chart.Visibility = Visibility.Visible;

                            pieSeriesList[i].Title.Text = food.Name;
                            double fillRate = Math.Round(food.FillRate, 2); // Yüzdelik değeri 2 ondalık basamakla yuvarla
                            double emptyRate = Math.Round(100 - fillRate, 2);

                            Image cricitalIcon = pieSeriesList[i].CriticalIcon;
                            Image warningIcon = pieSeriesList[i].WarningIcon;
                            Rectangle rect = pieSeriesList[i].Rectangle;

                            UpdateIconsAndRectangleVisibility(fillRate, cricitalIcon, warningIcon, rect);

                            // Güncelle
                            fillPie.Values[0] = fillRate;
                            emptyPie.Values[0] = emptyRate;
                            UpdatePieSeriesColor(fillPie, fillRate);
                            break;
                        }
                    }
                }

            }   

            // Boş Series'leri kontrol et ve PieChart'ı gizle
            foreach (var pair in pieSeriesList)
            {
                // Pair'in Title'ı varsa ve FoodList'te bulunuyorsa, görünür yap
                if (!string.IsNullOrEmpty(pair.Title.Text) &&
                    foodList.Any(food => food.Name == pair.Title.Text))
                {
                    pair.Chart.Visibility = Visibility.Visible; // PieChart'ı göster
                    pair.Title.Visibility = Visibility.Visible;

                }
                else
                {
                    pair.Chart.Visibility = Visibility.Collapsed; // PieChart'ı gizle
                    pair.Title.Visibility = Visibility.Collapsed;

                }
            }


        }

        double dummyFillRate = 100;
        private void GenerateDummyData()
        {
            // Rastgele bir doluluk oranı üret
            Random random = new Random();
            dummyFillRate -= 5;

            // İlk PieChart (PieSeries1) için doluluk oranını güncelle
            var fillPie = PieSeries2[1] as PieSeries; // "Fill Rate" için
            var emptyPie = PieSeries2[0] as PieSeries; // "Empty" için

            if (fillPie != null && emptyPie != null)
            {
                // Yüzdelik değerleri güncelle
                fillPie.Values[0] = dummyFillRate;
                emptyPie.Values[0] = 100 - dummyFillRate;

                // Rengi güncelle
                UpdatePieSeriesColor(fillPie, dummyFillRate);

                

                // Başlık güncelle
                PieTitle1.Text = $"Fill Rate: {dummyFillRate}%";
            }
        }

        private void PlayAlertSound()
        {
            // Ses dosyasının yolu
            string filePath = @"C:\Users\amete\source\repos\FoodTracking\FoodTracking\Sound\alert.mp3"; // Veya doğru tam yol

            try
            {
                // NAudio ile MP3 dosyasını çal
                using (var audioFile = new AudioFileReader(filePath)) // MP3 dosyasını oku
                using (var outputDevice = new WaveOutEvent()) // Çıkış cihazını ayarla
                {
                    outputDevice.Init(audioFile); // Ses kaynağını cihazla eşleştir
                    outputDevice.Play(); // Çalmaya başla

                    // Sesin bitmesini bekle
                    while (outputDevice.PlaybackState == PlaybackState.Playing)
                    {
                        Thread.Sleep(100); // Bekleme
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Hata: " + ex.Message);
            }
            finally
            {
                // Thread tamamlandığında flag'i false yapıyoruz
                isThreadRunning = false;
            }
        }

        private bool isThreadRunning = false;
        private void UpdateIconsAndRectangleVisibility(double fillRate, Image criticalIcon, Image warningIcon, Rectangle rectangle)
        {



            if (fillRate < 20)
            {

                if (!isThreadRunning)
                {
                    // Yeni bir thread başlatılacaksa önce flag'i true yapıyoruz
                    isThreadRunning = true;

                    // Thread başlatma
                    Thread playThread = new Thread(PlayAlertSound);
                    playThread.Start();
                }

                criticalIcon.Visibility = Visibility.Visible;
                rectangle.Visibility = Visibility.Visible;
                warningIcon.Visibility = Visibility.Collapsed; // Gereksizse ekleyebilirsiniz
            }
            else if (fillRate < 40)
            {
                warningIcon.Visibility = Visibility.Visible;
                criticalIcon.Visibility = Visibility.Collapsed;
                rectangle.Visibility = Visibility.Collapsed;
            }
            else
            {
                criticalIcon.Visibility = Visibility.Collapsed;
                warningIcon.Visibility = Visibility.Collapsed;
                rectangle.Visibility = Visibility.Collapsed;
            }
        }



        private void UpdatePieSeriesColor(PieSeries pieSeries, double fillRate)
        {
            // FillRate'i yüzde (0-1 aralığına) dönüştür
            double percentage = fillRate / 100.0;

            // Mavi (tam dolu) ile kırmızı (tam boş) arasında geçiş
            Color interpolatedColor = InterpolateColor(Color.FromRgb(0, 255, 255), Color.FromRgb(255, 0, 0), 1 - percentage);

            // PieSeries'in rengini güncelle
            pieSeries.Fill = new SolidColorBrush(interpolatedColor);
        }

        private Color InterpolateColor(Color startColor, Color endColor, double percentage)
        {
            byte r = (byte)(startColor.R + (endColor.R - startColor.R) * percentage);
            byte g = (byte)(startColor.G + (endColor.G - startColor.G) * percentage);
            byte b = (byte)(startColor.B + (endColor.B - startColor.B) * percentage);

            return Color.FromRgb(r, g, b);
        }



        public class DataItem
        {
            public string Information { get; set; }
            public string Date { get; set; }
        }

        public SeriesCollection PieSeries1 { get; set; }
        public SeriesCollection PieSeries2 { get; set; }
        public SeriesCollection PieSeries3 { get; set; }
        public SeriesCollection PieSeries4 { get; set; }
        public SeriesCollection PieSeries5 { get; set; }
        public SeriesCollection PieSeries6 { get; set; }
        public SeriesCollection PieSeries7 { get; set; }
        public SeriesCollection PieSeries8 { get; set; }

        // Add more properties as needed

        private SeriesCollection CreatePieSeries()
        {
            return new SeriesCollection
            {
                new PieSeries { Values = new ChartValues<double> { 0 }, Title = "Empty", Fill = new SolidColorBrush(Color.FromRgb(0xF8, 0xEA, 0xEA))},
                new PieSeries { Values = new ChartValues<double> { 100 }, Title = "Fill Rate", Fill = new SolidColorBrush(Color.FromRgb(0x00,0xFF ,0xFF))}
            };
        }

        private void GenerateDummyData_Click(object sender, RoutedEventArgs e)
        {
            GenerateDummyData();
        }

        private void PieChart1Slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (PieSeries1 == null || PieSeries1.Count < 2)
            {
                MessageBox.Show("PieSeries1 uygun şekilde başlatılmamış. Lütfen kontrol edin.");
                return;
            }

            // Slider değeri
            double fillRate = Math.Round(e.NewValue, 2);
            double emptyRate = 100 - fillRate;

            // PieChart1'in PieSeries'lerini al
            var fillPie = PieSeries1[1] as PieSeries; // "Fill Rate" için
            var emptyPie = PieSeries1[0] as PieSeries; // "Empty" için

            if (fillPie != null && emptyPie != null)
            {
                // Yüzdelik değerleri güncelle
                fillPie.Values[0] = fillRate;
                emptyPie.Values[0] = emptyRate;

                // Rengi güncelle
                UpdatePieSeriesColor(fillPie, fillRate);

                UpdateIconsAndRectangleVisibility(fillRate, CriticalIcon1, WarningIcon1, Rectangle1);


            }

        }

    }


}
