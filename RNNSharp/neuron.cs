
/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class SimpleLayer
    {
        public double[] cellOutput;
        public double[] previousCellOutput;
        public double[] er;
        public bool[] mask;
        public int size;

        public SimpleLayer(int m)
        {
            cellOutput = new double[m];
            previousCellOutput = new double[m];
            er = new double[m];
            mask = new bool[m];

            size = m;
        }
    }
}
