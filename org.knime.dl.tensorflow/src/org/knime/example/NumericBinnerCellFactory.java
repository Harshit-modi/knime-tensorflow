/*
 * ------------------------------------------------------------------------
  * This source code, its documentation and all appendant files
 * are protected by copyright law. All rights reserved.
 *
 * Copyright, 2008 - 2012
 * KNIME.com, Zurich, Switzerland
 *
 * You may not modify, publish, transmit, transfer or sell, reproduce,
 * create derivative works from, distribute, perform, display, or in
 * any way exploit any of the content, in whole or in part, except as
 * otherwise expressly permitted in writing by the copyright owner or
 * as specified in the license file distributed with this product.
 *
 * If you have any questions please contact the copyright holder:
 * website: www.knime.com
 * email: contact@knime.com
 * ---------------------------------------------------------------------
 */
package org.knime.example;

import java.util.List;

import org.knime.core.data.DataCell;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.DataRow;
import org.knime.core.data.DataType;
import org.knime.core.data.DoubleValue;
import org.knime.core.data.container.SingleCellFactory;
import org.knime.core.data.def.IntCell;

/**
 * 
 * @author KNIME.com AG, Zurich, Switzerland
 */
public class NumericBinnerCellFactory extends SingleCellFactory {

    private final List<Double> m_intervalUpperBounds;
    
    private final int m_colIndex;
    
    private final NumericBin[] m_bins;
    

    /**
     * Create new numeric binner factory.
     * @param newColSpec the column spec of the column with the binning 
     *  information.
     * @param intervalUpperBounds the upper interval bounds of the bins
     * @param columnIndex the selected column index.
     * @param bins the representation of the bins.
     */
    public NumericBinnerCellFactory(final DataColumnSpec newColSpec,
            final List<Double>intervalUpperBounds, final int columnIndex,
            final NumericBin[] bins) {
        super(newColSpec);
        if (intervalUpperBounds == null) {
            throw new NullPointerException("Interval bounds must not be null!");
        }
        m_intervalUpperBounds = intervalUpperBounds;
        m_colIndex = columnIndex;
        m_bins = bins;
        
    }
    
    /** {@inheritDoc} */
    @Override
    public DataCell getCell(final DataRow row) {
        DataCell currCell = row.getCell(m_colIndex);
        // check the cell for missing value
        if (currCell.isMissing()) {
            return DataType.getMissingCell();
        }
        double currValue = ((DoubleValue)currCell).getDoubleValue();
        int binNr = 0;
        for (Double intervalBound : m_intervalUpperBounds) {
            if (currValue <= intervalBound) {
                m_bins[binNr].addRowToBin(row.getKey());
                return new IntCell(binNr);
            }
            binNr++;
        }
        return DataType.getMissingCell();
    }
    
    /**
     * @return the filled bins.
     */
    public NumericBin[] getBins() {
        return m_bins;
    }

}
