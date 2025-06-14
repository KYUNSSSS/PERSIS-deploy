const express = require('express');
const router = express.Router();
const _ = require("lodash");

// -- Load Common Authentication -- //
const authenticateRoute = require('../common/authenticate');
// -- Load Controller -- //
const user = require("../controllers/user-controller");

// @route   POST api/user/login
// @desc    Find System Parameter
// @access  Private
router.post("/login", user.login);



// @route   POST api/user/change-password
// @desc    Update user Password
// @access  Public
router.post('/change_password', authenticateRoute, user.change_password);

// @route   POST api/user/create
// @desc    create System Parameter
// @access  Private
router.post("/create", user.create);




// @route   GET api/user/detail
// @desc    List System Parameter
// @access  Private
router.get("/detail", authenticateRoute, user.detail);

// @route   POST api/user/update
// @desc    update System Parameter
// @access  Private
router.post("/update", authenticateRoute, user.update);

// @route   POST api/user/logout
// @desc    Logout user
// @access  Private
router.post("/logout", authenticateRoute, user.logout);

module.exports = router;