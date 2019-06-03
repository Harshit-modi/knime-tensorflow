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

import java.awt.Rectangle;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Set;

import org.knime.core.data.RowKey;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.ModelContentRO;
import org.knime.core.node.ModelContentWO;

/**
 * A small model representing a bin with the rows contained in that bin and 
 * a graphical representation as a rectangle.
 * 
 * @author KNIME.com AG, Zurich, Switzerland
 */
public class NumericBin {
    
    private static final String CFG_KEY_CELLS = "rowIds";
    
    private final Set<RowKey> m_containedRowIds;
    
    private Rectangle m_viewRepresentation;
    
    private boolean m_isHilite;
    
    private boolean m_isSelected;
    
    /**
     * Creates a new numeric bin. 
     */
    public NumericBin() {
        m_containedRowIds = new LinkedHashSet<RowKey>();
    }
    
    /**
     * Adds another row to this bin.
     * @param rowId the row to add to this bin.
     */
    public void addRowToBin(final RowKey rowId) {
        m_containedRowIds.add(rowId);
    }
    
    /**
     * @return the number of rows in this bin.
     */
    public int getSize() {
        return m_containedRowIds.size();
    }
    
    /**
     * @return the ids of the rows in this bin.
     */
    public Set<RowKey> getRowKeys() {
        return m_containedRowIds;
    }
    
    /**
     * @return the graphical representation as a rectangle.
     */
    public Rectangle getViewRepresentation() {
        return m_viewRepresentation;
    }
    
    /**
     * The graphical representation can only be calculated outside with the
     * knowledge of the number of bins, the maximal and minimal size 
     * and the available width and height. This is done in the 
     * {@link NumericBinnerViewPanel#paint(java.awt.Graphics)}
     * 
     * @param rectangle the graphical representation
     */
    public void setViewRepresentation(final Rectangle rectangle) {
        m_viewRepresentation = rectangle;
    }
    
    // ************* loading and saving of internal representation ***********
    
    /**
     * Adds the IDs of the contained rows to the settings. That's enough
     * to later on restore the visual representation, since that only depends on
     * the dimension of the panel and the number of contained rows per bin.
     * 
     * @param modelContent the model content object to save to.
     */
    public void saveTo(final ModelContentWO modelContent) {
        RowKey[] cellArray = new RowKey[m_containedRowIds.size()]; 
        m_containedRowIds.toArray(cellArray);
        modelContent.addRowKeyArray(CFG_KEY_CELLS, cellArray);
    }
    
    /**
     * Loads the contained row IDs.
     *  
     * @param modelContent to read contained RowKeys from.
     * @throws InvalidSettingsException if the settings are invalid
     */
    public void loadFrom(final ModelContentRO modelContent) 
        throws InvalidSettingsException {
        RowKey[] rkArray = modelContent.getRowKeyArray(CFG_KEY_CELLS);
        m_containedRowIds.addAll(Arrays.asList(rkArray));
    }
    
    // ************* hilite support *****************

    /**
     * @param isHilite sets the hilite status of this bin.
     */
    public void setHilited(final boolean isHilite) {
        m_isHilite = isHilite;
    }
    
    /**
     * @return true if this bin contains hilited keys, false otherwise.
     */
    public boolean isHilited() {
        return  m_isHilite;
    }
    
    /**
     * @return true if this bin is selected. false otherwise.
     */
    public boolean isSelected() {
        return m_isSelected;
    }
    
    /**
     * @param selected true, if the bin is selected, false otherwise.
     */
    public void setSelected(final boolean selected) {
        m_isSelected = selected;
    }
    
}
