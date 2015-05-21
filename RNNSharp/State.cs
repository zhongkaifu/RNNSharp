using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
        SparseVector m_SparseData = new SparseVector();

        //Store dense features, such as word embedding
        Vector m_spDenseData = null;

        //Store run time features
        PriviousLabelFeature[] m_RuntimeFeatures;
        int m_NumRuntimeFeature;

        int m_Label;

        public int GetLabel() { return m_Label; }

        public SparseVector GetSparseData() { return m_SparseData; }

        public Vector GetDenseData() { return m_spDenseData; }


        public PriviousLabelFeature GetRuntimeFeature(int i) { return m_RuntimeFeatures[i]; }

        public int GetNumRuntimeFeature() { return m_NumRuntimeFeature; }


        public void SetNumRuntimeFeature(int n)
        {
            if (m_NumRuntimeFeature != n)
            {
                m_NumRuntimeFeature = n;
                m_RuntimeFeatures = null;
                if (m_NumRuntimeFeature > 0)
                    m_RuntimeFeatures = new PriviousLabelFeature[m_NumRuntimeFeature];
            }
        }


        public void SetRuntimeFeature(int i, int offset, double v)
        {
            PriviousLabelFeature f = m_RuntimeFeatures[i];
            m_SparseData.ChangeValue(f.PositionInSparseVector, f.StartInDimension + offset, v);
        }


        public void SetDenseData(Vector dense)
        {
            m_spDenseData = dense;
        }

        public void SetLabel(int label)
        {
            m_Label = label;
        }


        public int GetDenseDimension()
        {
            if (null != m_spDenseData)
                return m_spDenseData.GetDimension();
            else
                return 0;
        }

        public int GetSparseDimension()
        {
            return m_SparseData.GetDimension();
        }


        public void AddRuntimeFeaturePlacehold(int i, int offsetToCurentState, int posInSparseVector, int startInDimension)
        {
            PriviousLabelFeature r = new PriviousLabelFeature();
            r.OffsetToCurrentState = offsetToCurentState;
            r.StartInDimension = startInDimension;
            r.PositionInSparseVector = posInSparseVector;
            m_RuntimeFeatures[i] = r;
        }

    }
}
