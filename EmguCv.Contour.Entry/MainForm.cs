using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using EmguCv.Client;

namespace EmguCv.Contour.Entry
{
    public partial class MainForm : Form
    {
        //private static readonly string PicPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Samples", "粤BP121E.jpg"); //Y
        private static readonly string PicPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Samples", "粤A2AD89.jpg"); //Y
        //private static readonly string PicPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Samples", "粤B366RS.jpg"); //Y 偶然
        //private static readonly string PicPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Samples", "粤B868TA.jpg"); //Y 偶然
        //private static readonly string PicPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Samples", "粤B38822.jpg");

        public MainForm()
        {
            InitializeComponent();
        }

        private void btnRecognise_Click(object sender, EventArgs e)
        {
            var image = Graphy.Load(PicPath);
            pbSrc.Image = image.ToBitmap();

            var size = image.Size;

            var gray = Graphy.Gray(image);
            pbLocated.Image = gray.ToBitmap();

            var sobel = Graphy.Sobel(gray, size, 2, 0, 3);
            pbLocated.Image = Converter.ToBitmap<Gray, byte>(sobel);

            var converted = Graphy.ConvertScale(sobel, size, 0.00390625, 0);
            pbLocated.Image = Converter.ToBitmap<Gray, byte>(converted);

            var threshold = Graphy.Threshold(converted, size);
            pbLocated.Image = threshold.ToBitmap();

            var morphology = Morphology(threshold, size);
            pbLocated.Image = morphology.ToBitmap();

            var contours = GetContours(morphology);
            lblTotalContours.Text = string.Format("总矩形轮廓数:{0}", contours.Count);
            if (contours.Count == 0)
            {
                return;
            }

            contours = FiltrateByRatio(contours, 3, 6);
            lblFiltratedContours.Text = string.Format("筛选后轮廓数:{0}", contours.Count);
            if (contours.Count == 0)
            {
                return;
            }

            var copy = Operator.Copy(image);

            var marked = MarkContours(copy, contours);
            pbLocated.Image = marked.ToBitmap();

            var rawPlate = Operator.Cut(image, contours[0]);
            pbPlate.Image = rawPlate.ToBitmap();

            var thresholdRawPlate = GrayAndThreshold(rawPlate);
            pbPlate.Image = thresholdRawPlate.ToBitmap();

            var h = ProjectAxisY(thresholdRawPlate);

            var edgeY = GetEdgeY(image, h);

            var plate = Operator.Cut(rawPlate, new Rectangle(0, edgeY.Top, rawPlate.Width, edgeY.Bottom - edgeY.Top));
            pbPlate.Image = plate.ToBitmap();

            var thresholdPlate = GrayAndThreshold(plate);
            pbPlate.Image = thresholdPlate.ToBitmap();

            var v = new int[thresholdPlate.Width];
            for (var x = 0; x < thresholdPlate.Width; x++)
            {
                for (var y = 0; y < thresholdPlate.Height; y++)
                {
                    var s = CvInvoke.cvGet2D(thresholdPlate, y, x);
                    if (s.v0 == 255)
                    {
                        v[x]++;
                    }
                }
            }

            var edgeCount = 0;
            var edgeXs = new List<int>();

            for (var x = 0; x < thresholdPlate.Width - 1; x++)
            {
                var current = v[x];
                var next = v[x + 1];

                if ((current == 0 && next != 0) || (current != 0 && next == 0))
                {
                    edgeCount++;
                    edgeXs.Add(x);
                }
            }

            if (edgeCount == 20)
            {
                edgeXs = edgeXs.Skip(2).Take(16).ToList();
            }

            if (edgeCount == 18)
            {
                edgeXs = edgeXs.Skip(1).Take(16).ToList();
            }


            var paintX = CvInvoke.cvCreateImage(thresholdPlate.Size, IPL_DEPTH.IPL_DEPTH_8U, 1);
            CvInvoke.cvZero(paintX);

            for (var x = 0; x < image.Width; x++)
            {
                for (var y = 0; y < thresholdPlate.Height - v[x]; y++)
                {
                    var t = new MCvScalar(255);
                    CvInvoke.cvSet2D(paintX, y, x, t);
                }
            }

            pbProjectionX.Image = Converter.ToBitmap<Gray, byte>(paintX);

            var ocrChar = new Emgu.CV.OCR.Tesseract(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Tessdata"), "eng", Emgu.CV.OCR.Tesseract.OcrEngineMode.OEM_TESSERACT_ONLY);
            ocrChar.SetVariable("tessedit_char_whitelist", "ABCDEFGHJKLMNPQRSTUVWXYZ");

            var ocrCn = new Emgu.CV.OCR.Tesseract(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Tessdata"), "chi_sim", Emgu.CV.OCR.Tesseract.OcrEngineMode.OEM_TESSERACT_ONLY);
            ocrCn.SetVariable("tessedit_char_whitelist", "粤");

            var ocr = new Emgu.CV.OCR.Tesseract(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Tessdata"), "eng", Emgu.CV.OCR.Tesseract.OcrEngineMode.OEM_TESSERACT_ONLY);
            ocr.SetVariable("tessedit_char_whitelist", "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ");

            //第一个字符，中文
            var charImage1 = GrayAndThreshold2(Operator.Cut(plate, new Rectangle(edgeXs[0], 0, edgeXs[1] - edgeXs[0], plate.Height)));
            pbChar1.Image = charImage1.ToBitmap();

            //第二个字符，英文字母
            var charImage2 = GrayAndThreshold2(Operator.Cut(plate, new Rectangle(edgeXs[2], 0, edgeXs[3] - edgeXs[2], plate.Height)));
            pbChar2.Image = charImage2.ToBitmap();

            //一个点
            var dotImage = Operator.Cut(plate, new Rectangle(edgeXs[4], 0, edgeXs[5] - edgeXs[4], plate.Height));

            //第三个字符，英文或字母
            var charImage3 = GrayAndThreshold2(Operator.Cut(plate, new Rectangle(edgeXs[6], 0, edgeXs[7] - edgeXs[6], plate.Height)));
            pbChar3.Image = charImage3.ToBitmap();

            //第四个字符，英文或字母
            var charImage4 = GrayAndThreshold2(Operator.Cut(plate, new Rectangle(edgeXs[8], 0, edgeXs[9] - edgeXs[8], plate.Height)));
            pbChar4.Image = charImage4.ToBitmap();

            //第五个字符，英文或字母
            var charImage5 = GrayAndThreshold2(Operator.Cut(plate, new Rectangle(edgeXs[10], 0, edgeXs[11] - edgeXs[10], plate.Height)));
            pbChar5.Image = charImage5.ToBitmap();

            //第六个字符，英文或字母
            var charImage6 = GrayAndThreshold2(Operator.Cut(plate, new Rectangle(edgeXs[12], 0, edgeXs[13] - edgeXs[12], plate.Height)));
            pbChar6.Image = charImage6.ToBitmap();

            //第七个字符，英文或字母
            var charImage7 = GrayAndThreshold2(Operator.Cut(plate, new Rectangle(edgeXs[14], 0, edgeXs[15] - edgeXs[14], plate.Height)));
            pbChar7.Image = charImage7.ToBitmap();

            ocrCn.Recognize(charImage1);
            var char1 = ocrCn.GetText();
            char1 = char1.Replace(Environment.NewLine, string.Empty);
            lblChar1.Text = char1;

            ocrChar.Recognize(charImage2);
            var char2 = ocrChar.GetText();
            char2 = char2.Replace(Environment.NewLine, string.Empty);
            lblChar2.Text = char2;

            ocr.Recognize(charImage3);
            var char3 = ocr.GetText();
            char3 = char3.Replace(Environment.NewLine, string.Empty);
            lblChar3.Text = char3;

            ocr.Recognize(charImage4);
            var char4 = ocr.GetText();
            char4 = char4.Replace(Environment.NewLine, string.Empty);
            lblChar4.Text = char4;

            ocr.Recognize(charImage5);
            var char5 = ocr.GetText();
            char5 = char5.Replace(Environment.NewLine, string.Empty);
            lblChar5.Text = char5;

            ocr.Recognize(charImage6);
            var char6 = ocr.GetText();
            char6 = char6.Replace(Environment.NewLine, string.Empty);
            lblChar6.Text = char6;

            ocr.Recognize(charImage7);
            var char7 = ocr.GetText();
            char7 = char7.Replace(Environment.NewLine, string.Empty);
            lblChar7.Text = char7;

            lblPlate.Text = char1 + char2 + char3 + char4 + char5 + char6 + char7;
        }

        private static EdgeY GetEdgeY(CvArray<byte> image, IList<int> h)
        {
            var edgeY = new EdgeY();

            var min = int.MaxValue;
            for (var i = 0; i < image.Height / 2; i++)
            {
                if (h[i] <= min)
                {
                    min = h[i];
                    edgeY.Top = i;
                }
            }

            min = int.MaxValue;
            for (var i = image.Height - 1; i >= image.Height / 2; i--)
            {
                if (h[i] <= min)
                {
                    min = h[i];
                    edgeY.Bottom = i;
                }
            }
            return edgeY;
        }

        private Image<Gray, byte> Morphology(IntPtr image, Size size)
        {
            var morphology = new Image<Gray, Byte>(size);

            var structuring = CvInvoke.cvCreateImage(size, IPL_DEPTH.IPL_DEPTH_8U, 1);

            CvInvoke.cvCreateStructuringElementEx(3, 1, 1, 0, CV_ELEMENT_SHAPE.CV_SHAPE_RECT, structuring);
            CvInvoke.cvDilate(image, morphology, structuring, 6);
            CvInvoke.cvErode(morphology, morphology, structuring, 7);
            CvInvoke.cvDilate(morphology, morphology, structuring, 1);

            CvInvoke.cvCreateStructuringElementEx(1, 3, 0, 1, CV_ELEMENT_SHAPE.CV_SHAPE_RECT, structuring);
            CvInvoke.cvErode(morphology, morphology, structuring, 2);
            CvInvoke.cvDilate(morphology, morphology, structuring, 2);

            return Converter.ToEmguCvImage<Gray, byte>(morphology);
        }

        private List<Rectangle> GetContours(Image<Gray, byte> image)
        {
            using (var stor = new MemStorage())
            {
                var rectangles = new List<Rectangle>();
                var contours = image.FindContours(CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE, RETR_TYPE.CV_RETR_CCOMP, stor);

                while (contours != null)
                {
                    rectangles.Add(contours.BoundingRectangle);
                    contours = contours.HNext;
                }

                return rectangles;
            }
        }

        private static List<Rectangle> FiltrateByRatio(IEnumerable<Rectangle> contours, double ratioMin, double ratioMax)
        {
            return contours.Where(c =>
            {
                var ratio = c.Width / c.Height;
                return (ratioMin <= ratio && ratio <= ratioMax);
            }).ToList();
        }

        private Image<Bgr, byte> MarkContours(Image<Bgr, byte> image, IEnumerable<Rectangle> contours)
        {
            foreach (var contour in contours)
            {
                image.Draw(contour, new Bgr(Color.Red), 2);
            }

            return image;
        }

        private int[] ProjectAxisY(CvArray<byte> image)
        {
            var values = new int[image.Height];

            for (var y = 0; y < image.Height; y++)
            {
                for (var x = 0; x < image.Width; x++)
                {
                    var s = CvInvoke.cvGet2D(image, y, x);
                    if (s.v0 == 255)
                    {
                        values[y]++;
                    }
                }
            }

            return values;
        }

        private Image<Gray, byte> GrayAndThreshold(Image<Bgr, byte> image)
        {
            var gray = Graphy.Gray(image);
            var threshold = Graphy.Threshold(gray);
            return threshold;
        }

        private Image<Gray, byte> GrayAndThreshold2(Image<Bgr, byte> image)
        {
            var gray = Graphy.Gray(image);
            var threshold = Graphy.Threshold(gray);
            return threshold;
        }
    }

    public class EdgeY
    {
        public int Top { get; set; }
        public int Bottom { get; set; }
    }
}
