
/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class Matrix<T>
    {

        public int Height { get; set; } // the number of rows
        public int Width { get; set; } // the number of columns
        private T[][] m_saData;

        public Matrix(int h, int w)
        {
            Height = h;
            Width = w;
            m_saData = new T[Height][];
            for (int i = 0; i < Height; i++)
            {
                m_saData[i] = new T[Width];
            }
        }


        public T[] this[int i]
        {
            get
            {
                return m_saData[i];
            }
            set
            {
                m_saData[i] = value;
            }
        }

        public Matrix<T> CopyTo()
        {
            Matrix<T> m = new Matrix<T>(Height, Width);

            for (int i = 0; i < Height; i++)
            {
                m_saData[i].CopyTo(m[i], 0);
            }

            return m;
        }
    }
}
