using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace EmguCv.Client
{
    public class Operator
    {
        public static Image<Gray, byte> Cut(Image<Gray, byte> image, Rectangle rectangle)
        {
            var ptr = CvInvoke.cvCreateImage(new Size(rectangle.Width, rectangle.Height), IPL_DEPTH.IPL_DEPTH_8U, 3);
            CvInvoke.cvSetImageROI(image, rectangle);
            CvInvoke.cvCopy(image, ptr, IntPtr.Zero);

            return Converter.ToEmguCvImage<Gray, byte>(ptr);
        }

        public static Image<Bgr, byte> Cut(Image<Bgr, byte> image, Rectangle rectangle)
        {
            var ptr = CvInvoke.cvCreateImage(new Size(rectangle.Width, rectangle.Height), IPL_DEPTH.IPL_DEPTH_8U, 3);
            CvInvoke.cvSetImageROI(image, rectangle);
            CvInvoke.cvCopy(image, ptr, IntPtr.Zero);

            return Converter.ToEmguCvImage<Bgr, byte>(ptr);
        }

        public static Image<Bgr, byte> Copy(Image<Bgr, byte> image)
        {
            var copy = new Image<Bgr, byte>(image.Width, image.Height);
            image.CopyTo(copy);
            return copy;
        }
    }
}
