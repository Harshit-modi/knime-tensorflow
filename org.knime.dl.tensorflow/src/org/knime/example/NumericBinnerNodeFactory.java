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

import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeFactory;
import org.knime.core.node.NodeView;

/**
 * <code>NodeFactory</code> for the "NumericBinner" Node.
 * Simple numeric binner with equidistant bins.
 *
 * @author KNIME.com AG, Zurich, Switzerland
 */
public class NumericBinnerNodeFactory 
        extends NodeFactory<NumericBinnerNodeModel> {

    /** {@inheritDoc} */
    @Override
    public NumericBinnerNodeModel createNodeModel() {
        return new NumericBinnerNodeModel();
    }

    /** {@inheritDoc} */
    @Override
    public int getNrNodeViews() {
        return 1;
    }

    /** {@inheritDoc} */
    @Override
    public NodeView<NumericBinnerNodeModel> createNodeView(final int viewIndex,
            final NumericBinnerNodeModel nodeModel) {
        return new NumericBinnerNodeView(nodeModel);
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasDialog() {
        return true;
    }

    /** {@inheritDoc} */
    @Override
    public NodeDialogPane createNodeDialogPane() {
        return new NumericBinnerNodeDialog();
    }

}
