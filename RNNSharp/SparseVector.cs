using System.Collections.Generic;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class SparseVector : SingleVector
    {
        List<KeyValuePair<int, float>> kvData;
        int length;

        public override int Length { get { return length; } }

        public IEnumerator<KeyValuePair<int, float>> GetEnumerator()
        {
            foreach (KeyValuePair<int, float> pair in kvData)
            {
                yield return pair;
            }
        }


        public SparseVector()
        {
            kvData = new List<KeyValuePair<int, float>>();
        }

        public void ChangeValue(int positionInSparseVector, int dimension, float value)
        {
            kvData[positionInSparseVector] = new KeyValuePair<int, float>(dimension, value);
        }

        public void SetLength(int len) { length = len; }

        public void AddKeyValuePairData(Dictionary<int, float> kv)
        {
            foreach (KeyValuePair<int, float> pair in kv)
            {
                kvData.Add(pair);
            }
        }

        public void AddKeyValuePairData(SparseVector sparseVector)
        {
            foreach (KeyValuePair<int, float> pair in sparseVector)
            {
                kvData.Add(pair);
            }
        }
    }
}
