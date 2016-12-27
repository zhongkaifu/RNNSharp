namespace RNNSharp
{
    public class SequencePair : ISequence
    {
        public RNNDecoder autoEncoder = null;

        public Sentence srcSentence;

        //     public Sequence srcSequence;
        public Sequence tgtSequence;

        int ISequence.DenseFeatureSize => tgtSequence.DenseFeatureSize;

        int ISequence.SparseFeatureSize => tgtSequence.SparseFeatureSize;
    }
}