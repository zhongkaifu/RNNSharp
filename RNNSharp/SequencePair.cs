using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp
{
    public class SequencePair : ISequence
    {
   //     public Sequence srcSequence;
        public Sequence tgtSequence;

        public Sentence srcSentence;
        public RNNSharp.RNNDecoder autoEncoder = null;

        int ISequence.DenseFeatureSize
        {
            get
            {
                return tgtSequence.DenseFeatureSize;
            }
        }

        int ISequence.SparseFeatureSize
        {
            get
            {
                return tgtSequence.SparseFeatureSize;
            }
        }
    }
}
