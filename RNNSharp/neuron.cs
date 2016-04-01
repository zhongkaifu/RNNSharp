
/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class SimpleLayer
    {
        public double[] cellOutput;
        public double[] er;
        public bool[] mask;
        public int size;

        public SimpleLayer(int m)
        {
            cellOutput = new double[m];
            er = new double[m];
            mask = new bool[m];

            size = m;
        }
    }
}
