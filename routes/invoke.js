const express = require('express');
const router = express.Router();
const _ = require("lodash");

// -- Load Common Authentication -- //
const authenticateRoute = require('../common/authenticate');
// -- Load Controller -- //
const invoke = require("../controllers/invoke-controller");





// @route   POST api/invoke/runPythonProcess
// @desc    Run Python Process
// @access  Public
router.post('/runPythonProcess', authenticateRoute, invoke.runPythonProcess);


module.exports = router;