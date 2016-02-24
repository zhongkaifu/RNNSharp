
/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class PriviousLabelFeature
    {
        public int OffsetToCurrentState;
        public int StartInDimension;
        public int PositionInSparseVector;
    }

    public class State
    {
        //Store sparse features, such as template features
        public SparseVector SparseData { get;}
        //Store dense features, such as word embedding
        public VectorBase DenseData { get; set; }
        public int Label { get; set; }
        //Store run time features
        public PriviousLabelFeature[] RuntimeFeatures { get; set; }

        public State()
        {
            SparseData = new SparseVector();
        }

        public void SetRuntimeFeature(int i, int offset, float v)
        {
            PriviousLabelFeature f = RuntimeFeatures[i];
            SparseData.ChangeValue(f.PositionInSparseVector, f.StartInDimension + offset, v);
        }

        public void AddRuntimeFeaturePlacehold(int i, int offsetToCurentState, int posInSparseVector, int startInDimension)
        {
            PriviousLabelFeature r = new PriviousLabelFeature();
            r.OffsetToCurrentState = offsetToCurentState;
            r.StartInDimension = startInDimension;
            r.PositionInSparseVector = posInSparseVector;
            RuntimeFeatures[i] = r;
        }

    }
}
