using System;
using System.Collections.Generic;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace RNNSharp
{
    public class DataSet<T> where T : ISequence
    {
        public DataSet(int tagSize)
        {
            TagSize = tagSize;
            SequenceList = new List<T>();
            CRFLabelBigramTransition = new List<List<float>>();
        }

        public List<T> SequenceList { get; set; }
        public int TagSize { get; set; }
        public List<List<float>> CRFLabelBigramTransition { get; set; }

        public int DenseFeatureSize => 0 == SequenceList.Count ? 0 : SequenceList[0].DenseFeatureSize;

        public int SparseFeatureSize => 0 == SequenceList.Count ? 0 : SequenceList[0].SparseFeatureSize;

        public void Shuffle()
        {
            var rnd = new Random(DateTime.Now.Millisecond);
            for (var i = 0; i < SequenceList.Count; i++)
            {
                var m = rnd.Next() % SequenceList.Count;
                var tmp = SequenceList[i];
                SequenceList[i] = SequenceList[m];
                SequenceList[m] = tmp;
            }
        }

        public void BuildLabelBigramTransition(float smooth = 1.0f)
        {
            CRFLabelBigramTransition = new List<List<float>>();

            for (var i = 0; i < TagSize; i++)
            {
                CRFLabelBigramTransition.Add(new List<float>());
            }
            for (var i = 0; i < TagSize; i++)
            {
                for (var j = 0; j < TagSize; j++)
                {
                    CRFLabelBigramTransition[i].Add(smooth);
                }
            }

            for (var i = 0; i < SequenceList.Count; i++)
            {
                var sequence = SequenceList[i] as Sequence;
                if (sequence.States.Length <= 1)
                    continue;

                var pLabel = sequence.States[0].Label;
                for (var j = 1; j < sequence.States.Length; j++)
                {
                    var label = sequence.States[j].Label;
                    CRFLabelBigramTransition[label][pLabel]++;
                    pLabel = label;
                }
            }

            for (var i = 0; i < TagSize; i++)
            {
                float sum = 0;
                for (var j = 0; j < TagSize; j++)
                {
                    sum += CRFLabelBigramTransition[i][j];
                }

                for (var j = 0; j < TagSize; j++)
                {
                    CRFLabelBigramTransition[i][j] = (float)Math.Log(CRFLabelBigramTransition[i][j] / sum);
                }
            }
        }
    }
}