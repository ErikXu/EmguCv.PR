using System;
using System.Drawing;
using System.Runtime.InteropServices;
using Emgu.CV;
using Emgu.CV.Structure;

namespace EmguCv.Client
{
    public static class Converter
    {
        public static Image<TColor, TDepth> ToEmguCvImage<TColor, TDepth>(IntPtr ptr)
            where TColor : struct, IColor
            where TDepth : new()
        {
            var mi = (MIplImage)Marshal.PtrToStructure(ptr, typeof(MIplImage));
            return new Image<TColor, TDepth>(mi.width, mi.height, mi.widthStep, mi.imageData);
        }

        public static Bitmap ToBitmap<TColor, TDepth>(IntPtr ptr)
            where TColor : struct, IColor
            where TDepth : new()
        {
            var image = ToEmguCvImage<TColor, TDepth>(ptr);
            return image.ToBitmap();
        }
    }
}
