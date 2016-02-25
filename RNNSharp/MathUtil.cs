using System;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    class MathUtil
    {
        public static int GetMaxProbIndex(float [] array)
        {
            int dim = array.Length;
            double maxValue = array[0];
            int maxIdx = 0;
            for (int i = 1; i < dim; i++)
            {
                if (array[i] > maxValue)
                {
                    maxIdx = i;
                    maxValue = array[i];
                }
            }

            return maxIdx;
        }

         public const int MINUS_LOG_EPSILON = 13;
         public static double logsumexp(double x, double y, bool flg)
         {
             if (flg) return y;  // init mode
             double vmin = Math.Min(x, y);
             double vmax = Math.Max(x, y);
             if (vmax > vmin + MINUS_LOG_EPSILON)
             {
                 return vmax;
             }
             else
             {
                 return vmax + Math.Log(Math.Exp(vmin - vmax) + 1.0);
             }
         }
    }
}
