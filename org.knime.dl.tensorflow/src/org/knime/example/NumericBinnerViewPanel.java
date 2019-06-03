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

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Rectangle;

import javax.swing.JPanel;

import org.knime.core.data.property.ColorAttr;

/**
 * Draws the bins in the paint method.
 * 
 * @author KNIME.com AG, Zurich, Switzerland
 */
public class NumericBinnerViewPanel extends JPanel {
    
    // the bin representation
    private NumericBin[] m_bins;
    
    // initial size, if we have no size from the component
    private static final int SIZE = 600;
    
    /**
     * Sets the bins and the initial size.
     * @param bins the bins to draw.
     */
    public NumericBinnerViewPanel(final NumericBin[] bins) {
        m_bins = bins;
        setPreferredSize(new Dimension(SIZE, SIZE));
    }
    
    /**
     * If the view is updated the new bins are set and then painted.
     * 
     * @param bins the new bins to display.
     */
    public void updateView(final NumericBin[] bins) {
        m_bins = bins;
        repaint();
    }

    /** {@inheritDoc} */
    @Override
    public void paint(final Graphics g) {
        super.paint(g);
        if (m_bins != null && m_bins.length > 0) {
            int maxNr = 0;
            // determine the largest bin
            for (int i = 0; i < m_bins.length; i++) {
                maxNr = Math.max(m_bins[i].getSize(), maxNr);
            }
            // if no size information available (creation) set default size
            int width = getWidth();
            if (width == 0) {
                width = SIZE;
            }
            int height = getHeight();
            if (height == 0) {
                height = SIZE;
            }
            // calculate the bin width
            int binWidth = width / m_bins.length;
            for (int i = 0; i < m_bins.length; i++) {
                // the left side of the rectangle
                int x = i * binWidth;
                // the height of the bin
                int binHeight = height;
                // the larger the bin the higher the rect
                double sizeFactor = ((double)(maxNr - m_bins[i].getSize())
                        / (double)maxNr); 
                // since y-axis starts on top subtract 
                binHeight -= sizeFactor * height;
                Rectangle rect = new Rectangle(x, height - binHeight, binWidth, 
                        binHeight);
                m_bins[i].setViewRepresentation(rect);
                // draw a border in white to make the bins distinguishable
                Color color = Color.BLACK;
                if (m_bins[i].isHilited()) {
                    color = ColorAttr.HILITE;
                }
                if (m_bins[i].isSelected()) {
                    color = ColorAttr.SELECTED;
                }
                if (m_bins[i].isHilited() && m_bins[i].isSelected()) {
                    color = ColorAttr.SELECTED_HILITE;
                }
                Graphics2D g2 = (Graphics2D)g;
                g2.setColor(color);
                g2.fillRect(rect.x + 2, rect.y + 2, 
                        rect.width - 2, rect.height - 2);
                g2.setColor(Color.WHITE);
                g2.setStroke(new BasicStroke(2));
                g2.drawRect(rect.x, rect.y, rect.width, rect.height);
            }
        }
    }
    
    
    /**
     * 
     * @return all bins.
     */
    public NumericBin[] getBins() {
        return m_bins;
    }
}
