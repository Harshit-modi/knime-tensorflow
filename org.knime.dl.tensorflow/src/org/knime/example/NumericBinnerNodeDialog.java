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

import org.knime.core.data.DoubleValue;
import org.knime.core.node.defaultnodesettings.DefaultNodeSettingsPane;
import org.knime.core.node.defaultnodesettings.DialogComponentColumnNameSelection;
import org.knime.core.node.defaultnodesettings.DialogComponentNumber;
import org.knime.core.node.defaultnodesettings.SettingsModelIntegerBounded;
import org.knime.core.node.defaultnodesettings.SettingsModelString;

/**
 * <code>NodeDialog</code> for the "NumericBinner" Node.
 * Simple numeric binner with equidistant bins
 *
 * This node dialog derives from {@link DefaultNodeSettingsPane} which allows
 * creation of a simple dialog with standard components. If you need a more 
 * complex dialog please derive directly from 
 * {@link org.knime.core.node.NodeDialogPane}.
 * 
 * @author KNIME.com AG, Zurich, Switzerland
 */
public class NumericBinnerNodeDialog extends DefaultNodeSettingsPane {

    /**
     * New pane for configuring NumericBinner node dialog.
     * Contains control elements to adjust the number of bins 
     * and to select the column to bin.
     * Suppress warnings here: it is unavoidable since the 
     * allowed types passed as an generic array. 
     */
    @SuppressWarnings ("unchecked")
    protected NumericBinnerNodeDialog() {
        super();
        // nr of bins control element
        addDialogComponent(new DialogComponentNumber(
                new SettingsModelIntegerBounded(
                    NumericBinnerNodeModel.CFGKEY_NR_OF_BINS,
                    NumericBinnerNodeModel.DEFAULT_NR_OF_BINS,
                    1, Integer.MAX_VALUE),
                    "Number of bins:", /*step*/ 1));
        // column to bin
        addDialogComponent(new DialogComponentColumnNameSelection(
                new SettingsModelString(
                    NumericBinnerNodeModel.CFGKEY_COLUMN_NAME,
                    "Select a column"),
                    "Select the column to bin",
                    NumericBinnerNodeModel.IN_PORT,
                    DoubleValue.class));                    
    }
}
