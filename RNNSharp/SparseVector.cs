using System.Collections.Generic;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace RNNSharp
{
    public class SparseVector : SingleVector
    {
        private readonly List<KeyValuePair<int, float>> kvData;
        private int length;

        public SparseVector()
        {
            kvData = new List<KeyValuePair<int, float>>();
        }

        public override int Length => length;

        public IEnumerator<KeyValuePair<int, float>> GetEnumerator()
        {
            return ((IEnumerable<KeyValuePair<int, float>>)kvData).GetEnumerator();
        }

        public void ChangeValue(int positionInSparseVector, int dimension, float value)
        {
            kvData[positionInSparseVector] = new KeyValuePair<int, float>(dimension, value);
        }

        public void SetLength(int len)
        {
            length = len;
        }

        public void AddKeyValuePairData(Dictionary<int, float> kv)
        {
            foreach (var pair in kv)
            {
                kvData.Add(pair);
            }
        }

        public void AddKeyValuePairData(SparseVector sparseVector)
        {
            foreach (var pair in sparseVector)
            {
                kvData.Add(pair);
            }
        }
    }
}