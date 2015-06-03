using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace EmguCv.Client
{
    public static class Graphy
    {
        public static Image<Bgr, byte> Load(string imagePath)
        {
            return new Image<Bgr, byte>(imagePath);
        }

        /// <summary>
        /// 灰度化
        /// </summary>
        public static Image<Gray, byte> Gray(Image<Bgr, byte> image)
        {
            var ptr = CvInvoke.cvCreateImage(image.Size, IPL_DEPTH.IPL_DEPTH_8U, 1);
            CvInvoke.cvCvtColor(image, ptr, COLOR_CONVERSION.BGR2GRAY);
            return Converter.ToEmguCvImage<Gray, byte>(ptr);
        }

        /// <summary>
        /// 灰度均衡
        /// </summary>
        public static IntPtr GrayEqualize(IntPtr image, Size size)
        {
            var ptr = CvInvoke.cvCreateImage(size, IPL_DEPTH.IPL_DEPTH_8U, 1);
            CvInvoke.cvEqualizeHist(image, ptr);
            return ptr;
        }

        /// <summary>
        /// Sobel运算
        /// </summary>
        public static IntPtr Sobel(IntPtr image, Size size, int xOrder, int yOrder, int apertureSize)
        {
            var ptr = CvInvoke.cvCreateImage(size, IPL_DEPTH.IPL_DEPTH_8U, 1);
            CvInvoke.cvSobel(image, ptr, xOrder, yOrder, apertureSize);
            return ptr;
        }

        /// <summary>
        /// Sobel运算
        /// </summary>
        public static Image<Gray, byte> Sobel(Image<Gray, byte> image, int xOrder, int yOrder, int apertureSize)
        {
            var ptr = Sobel(image, image.Size, xOrder, yOrder, apertureSize);
            return Converter.ToEmguCvImage<Gray, byte>(ptr);
        }

        /// <summary>
        /// Sobel运算，x轴导数
        /// </summary>
        public static IntPtr SobelX(IntPtr image, Size size)
        {
            return Sobel(image, size, 1, 0, 3);
        }

        /// <summary>
        /// Sobel运算，x轴导数
        /// </summary>
        public static Image<Gray, byte> SobelX(Image<Gray, byte> image)
        {
            return Sobel(image, 1, 0, 3);
        }

        /// <summary>
        /// Sobel运算，y轴导数
        /// </summary>
        public static IntPtr SobelY(IntPtr image, Size size)
        {
            return Sobel(image, size, 1, 0, 3);
        }

        /// <summary>
        /// Sobel运算，y轴导数
        /// </summary>
        public static Image<Gray, byte> SobelY(Image<Gray, byte> image)
        {
            return Sobel(image, 0, 1, 3);
        }

        /// <summary>
        /// Sobel运算，x轴导数，绝对值
        /// </summary>
        public static IntPtr SobelAbsX(IntPtr image, Size size, double scale = 1d, double shift = 0d)
        {
            var sobelX = SobelX(image, size);
            var absX = CvInvoke.cvCreateImage(size, IPL_DEPTH.IPL_DEPTH_8U, 1);
            CvInvoke.cvConvertScaleAbs(sobelX, absX, scale, shift);
            return absX;
        }

        /// <summary>
        /// Sobel运算，x轴导数，绝对值
        /// </summary>
        public static Image<Gray, byte> SobelAbsX(Image<Gray, byte> image, double scale = 1d, double shift = 0d)
        {
            var absX = SobelAbsX(image, image.Size, scale, shift);
            return Converter.ToEmguCvImage<Gray, byte>(absX);
        }

        /// <summary>
        /// Sobel运算，y轴导数，绝对值
        /// </summary>
        public static IntPtr SobelAbsY(IntPtr image, Size size, double scale = 1d, double shift = 0d)
        {
            var sobelY = SobelY(image, size);
            var absY = CvInvoke.cvCreateImage(size, IPL_DEPTH.IPL_DEPTH_8U, 1);
            CvInvoke.cvConvertScaleAbs(sobelY, absY, scale, shift);
            return absY;
        }

        /// <summary>
        /// Sobel运算，y轴导数，绝对值
        /// </summary>
        public static Image<Gray, byte> SobelAbsY(Image<Gray, byte> image, double scale = 1d, double shift = 0d)
        {
            var absY = SobelAbsY(image, image.Size, scale, shift);
            return Converter.ToEmguCvImage<Gray, byte>(absY);
        }

        /// <summary>
        /// Sobel量子边缘检测
        /// </summary>
        public static IntPtr SobelEdgeDetect(IntPtr image, Size size, double scale = 1d, double shift = 0d, double alpha = 0.5, double beta = 0.5, double gamma = 0d)
        {
            var absX = SobelAbsX(image, size, scale, shift);
            var absY = SobelAbsY(image, size, scale, shift);

            var grad = CvInvoke.cvCreateImage(size, IPL_DEPTH.IPL_DEPTH_8U, 1);
            CvInvoke.cvAddWeighted(absX, alpha, absY, beta, gamma, grad);
            return grad;
        }

        /// <summary>
        /// Sobel量子边缘检测
        /// </summary>
        public static Image<Gray, byte> SobelEdgeDetect(Image<Gray, byte> image, double scale = 1d, double shift = 0d, double alpha = 0.5, double beta = 0.5, double gamma = 0d)
        {
            var grad = SobelEdgeDetect(image, image.Size, scale, shift, alpha, beta, gamma);
            return Converter.ToEmguCvImage<Gray, byte>(grad);
        }

        /// <summary>
        /// 二值化
        /// </summary>
        public static Image<Gray, byte> Threshold(IntPtr image, Size size)
        {
            var ptr = new Image<Gray, byte>(size);
            CvInvoke.cvThreshold(image, ptr, 0, 255, THRESH.CV_THRESH_OTSU);
            return Converter.ToEmguCvImage<Gray, byte>(ptr);
        }

        /// <summary>
        /// 二值化
        /// </summary>
        public static Image<Gray, byte> Threshold(Image<Gray, byte> image)
        {
            return Threshold(image, image.Size);
        }

        public static IntPtr ConvertScale(IntPtr image, Size size, double scale, double shift)
        {
            var ptr = CvInvoke.cvCreateImage(size, IPL_DEPTH.IPL_DEPTH_8U, 1);
            CvInvoke.cvConvertScale(image, ptr, scale, shift);
            return ptr;
        }

        public static Image<Gray, byte> ConvertScale(Image<Gray, byte> image, double scale, double shift)
        {
            var ptr = ConvertScale(image, image.Size, scale, shift);
            return Converter.ToEmguCvImage<Gray, byte>(ptr);
        }

        public static IntPtr GaussianBlur(IntPtr image, Size size, int param1, int param2, double param3, double param4)
        {
            var ptr = CvInvoke.cvCreateImage(size, IPL_DEPTH.IPL_DEPTH_8U, 1);
            CvInvoke.cvSmooth(image, ptr, SMOOTH_TYPE.CV_GAUSSIAN, param1, param2, param3, param4);
            return ptr;
        }

    }
}
