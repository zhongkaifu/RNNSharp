/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace RNNSharp
{
    public class PriviousLabelFeature
    {
        public int OffsetToCurrentState;
        public int PositionInSparseVector;
        public int StartInDimension;
    }

    public class State
    {
        public State()
        {
            SparseFeature = new SparseVector();
        }

        //Store sparse features, such as template features
        public SparseVector SparseFeature { get; }

        //Store dense features, such as word embedding
        public VectorBase DenseFeature { get; set; }

        public int Label { get; set; }

        //Store run time features
        public PriviousLabelFeature[] RuntimeFeatures { get; set; }

        public void SetRuntimeFeature(int i, int offset, float v)
        {
            var f = RuntimeFeatures[i];
            SparseFeature.ChangeValue(f.PositionInSparseVector, f.StartInDimension + offset, v);
        }

        public void AddRuntimeFeaturePlacehold(int i, int offsetToCurentState, int posInSparseVector,
            int startInDimension)
        {
            var r = new PriviousLabelFeature
            {
                OffsetToCurrentState = offsetToCurentState,
                StartInDimension = startInDimension,
                PositionInSparseVector = posInSparseVector
            };
            RuntimeFeatures[i] = r;
        }
    }
}