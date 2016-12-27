using System;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace RNNSharp
{
    internal class MathUtil
    {
        public const int MINUS_LOG_EPSILON = 13;

        public static int GetMaxProbIndex(float[] array)
        {
            var dim = array.Length;
            var maxValue = array[0];
            var maxIdx = 0;
            for (var i = 1; i < dim; i++)
            {
                if (array[i] > maxValue)
                {
                    maxIdx = i;
                    maxValue = array[i];
                }
            }

            return maxIdx;
        }

        public static double logsumexp(double x, double y, bool flg)
        {
            if (flg) return y; // init mode
            var vmin = Math.Min(x, y);
            var vmax = Math.Max(x, y);
            if (vmax > vmin + MINUS_LOG_EPSILON)
            {
                return vmax;
            }
            return vmax + Math.Log(Math.Exp(vmin - vmax) + 1.0);
        }
    }
}