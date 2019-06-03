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

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.LinkedHashSet;
import java.util.Set;

import javax.swing.JMenu;
import javax.swing.JMenuItem;

import org.knime.core.data.RowKey;
import org.knime.core.node.NodeView;
import org.knime.core.node.property.hilite.HiLiteHandler;
import org.knime.core.node.property.hilite.HiLiteListener;
import org.knime.core.node.property.hilite.KeyEvent;

/**
 * <code>NodeView</code> for the "NumericBinner" Node.
 * 
 * @author KNIME.com AG, Zurich, Switzerland
 */
public class NumericBinnerNodeView extends NodeView<NumericBinnerNodeModel> 
        implements HiLiteListener {
    
    private final JMenuItem m_hilite;
    
    private final JMenuItem m_unhilite;
    
    private final Set<NumericBin> m_selected;
    
    // panel which actually paints the bins
    private final NumericBinnerViewPanel m_panel;
    
    /**
     * Creates a new view.
     * 
     * @param nodeModel the model class: {@link NumericBinnerNodeModel}
     */
    protected NumericBinnerNodeView(final NumericBinnerNodeModel nodeModel) {
        super(nodeModel);
        // init empty panel, #modelChanged() will add bins
        m_panel = new NumericBinnerViewPanel(new NumericBin[0]);
        // sets the view content in the node view
        setComponent(m_panel);
        
        //************* mouse listener for selection of a bin *****************
        
        // add a mouse listener in order to determine the selected bins
        m_selected = new LinkedHashSet<NumericBin>();
        m_panel.addMouseListener(new MouseAdapter() {
            /** {@inheritDoc} */
            @Override
            public void mouseReleased(final MouseEvent e) {
                if (!e.isControlDown()) {
                    m_selected.clear();
                    for (NumericBin bin : m_panel.getBins()) {
                        bin.setSelected(false);
                    }
                }
                for (NumericBin bin : m_panel.getBins()) {
                    if (bin.getViewRepresentation() != null 
                            && bin.getViewRepresentation().contains(
                                    e.getX(), e.getY())) {
                        bin.setSelected(true);
                        m_selected.add(bin);
                        break;
                    }
                }
                m_panel.repaint();
            }
        });
        
        //************** the hilite menu **************************
        
        // create the hilite menu 
        // the HiliteHandler provides standard names 
        m_hilite = new JMenuItem(HiLiteHandler.HILITE_SELECTED);
        m_hilite.addActionListener(new ActionListener() {
            /** {@inheritDoc} */
            @Override
            public void actionPerformed(final ActionEvent e) {
                final Set<RowKey> toBeHilited = new LinkedHashSet<RowKey>();
                for (NumericBin bin : m_selected) {
                    // store all row ids from the selected bin
                    toBeHilited.addAll(bin.getRowKeys());
                }
                // now get the hilite handler and hilite the rows
                m_hiliteHandler.fireHiLiteEvent(
                        new KeyEvent(this, toBeHilited));
            }
            
        });
        m_unhilite = new JMenuItem(HiLiteHandler.UNHILITE_SELECTED);
        m_unhilite.addActionListener(new ActionListener() {
            /** {@inheritDoc} */
            @Override
            public void actionPerformed(final ActionEvent e) {
                Set<RowKey> toBeUnhilited = new LinkedHashSet<RowKey>();
                for (NumericBin bin : m_selected) {
                    // store all row ids that should be unhilited
                    toBeUnhilited.addAll(bin.getRowKeys());
                }
                // get the hilite handler and unhilite the rows
                m_hiliteHandler.fireUnHiLiteEvent(
                        new KeyEvent(this, toBeUnhilited));
            }
            
        });
        
        JMenuItem clear = new JMenuItem(HiLiteHandler.CLEAR_HILITE);
        clear.addActionListener(new ActionListener() {
            /** {@inheritDoc} */
            @Override
            public void actionPerformed(final ActionEvent e) {
                // get the hilite handler and unhilite all
                m_hiliteHandler.fireClearHiLiteEvent();
            } 
        });
        // create the menu and all the menu items to it
        JMenu menu = new JMenu(HiLiteHandler.HILITE);
        menu.add(m_hilite);
        menu.add(m_unhilite);
        menu.add(clear);
        // get the JMenu bar of the NodeView and add this menu to it
        getJMenuBar().add(menu);        
    }
    
    private HiLiteHandler m_hiliteHandler = null;

    /** {@inheritDoc} */
    @Override
    protected void modelChanged() {
        // update internal hilite handler
        HiLiteHandler hiliteHandler = getNodeModel().getInHiLiteHandler(0);
        if (m_hiliteHandler == null) {
            m_hiliteHandler = hiliteHandler;
            m_hiliteHandler.addHiLiteListener(this);
        } else {
            if (hiliteHandler != m_hiliteHandler) {
                m_hiliteHandler.removeHiLiteListener(this);
                m_hiliteHandler = hiliteHandler;
                m_hiliteHandler.addHiLiteListener(this);
            }
        }
        // if the model had changed get the new bins
        NumericBin[] bins = getNodeModel().getBinRepresentations();
        if (bins != null && bins.length > 0 && m_panel != null) {
            // and paint the bins
            m_panel.updateView(bins);
        } else {
            m_panel.updateView(new NumericBin[0]);
        }
    }

    /** {@inheritDoc} */
    @Override
    protected void onClose() {
        if (m_hiliteHandler != null) {
            m_hiliteHandler.removeHiLiteListener(this);
            m_hiliteHandler = null;
        }
    }

    /** {@inheritDoc} */
    @Override
    protected void onOpen() {
        // no op
    }

    /** {@inheritDoc} */
    @Override    
    public void hiLite(final KeyEvent event) {
        final Set<RowKey> hiliteKeys = m_hiliteHandler.getHiLitKeys();
        for (NumericBin bin : m_panel.getBins()) {
            // only hilite bin if all contained pattern are hilit
            if (hiliteKeys.containsAll(bin.getRowKeys())) {
                bin.setHilited(true);
            }
        }
        // and repaint to have the hilited bins displayed correctly
        m_panel.repaint();
    }

    /** {@inheritDoc} */
    @Override
    public void unHiLite(final KeyEvent event) {
        final Set<RowKey> hiliteKeys = m_hiliteHandler.getHiLitKeys();
        for (NumericBin bin : m_panel.getBins()) {
            final Set<RowKey> copyKeys = new LinkedHashSet<RowKey>(
                    bin.getRowKeys());
            // unhilite bin if at least one contained pattern is not hilit
            if (copyKeys.retainAll(hiliteKeys)) {
                bin.setHilited(false);
            }
        }
        // and repaint to have the hilited bins displayed correctly
        m_panel.repaint();
    }

    /** {@inheritDoc} */
    @Override
    public void unHiLiteAll(final KeyEvent event) {
        for (NumericBin bin : m_panel.getBins()) {
             bin.setHilited(false);
        }
        // repaint to display the bins correctly
        m_panel.repaint();
    }

}
