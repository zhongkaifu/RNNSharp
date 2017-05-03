namespace RNNSharp
{
    public enum TFEATURE_WEIGHT_TYPE_ENUM
    {
        BINARY,
        FREQUENCY
    }

    public enum LayerType
    {
        Softmax,
        SampledSoftmax,
        LSTM,
        DropOut,
        Simple,
        None
    }

    public enum PRETRAIN_TYPE
    {
        AutoEncoder,
        Embedding,
        None
    }

    public enum NETWORKTYPE
    {
        Forward = 0,
        ForwardSeq2Seq,
        BiDirectional,
        BiDirectionalAverage
    }

    public enum LAYERTYPE
    {
        BPTT = 0,
        LSTM
    }

    public enum RunningMode
    {
        Training = 0,
        Validate = 1,
        Test = 2
    }
}