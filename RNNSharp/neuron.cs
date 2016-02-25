
/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class SimpleCell
    {
        //cell output
        public float cellOutput;
        public float er;
        public bool mask;
    }

    public class SimpleLayer
    {
        public float[] cellOutput;
        public float[] er;
        public bool[] mask;
        public int size;

        public SimpleLayer(int m)
        {
            cellOutput = new float[m];
            er = new float[m];
            mask = new bool[m];

            size = m;
        }
    }
}
