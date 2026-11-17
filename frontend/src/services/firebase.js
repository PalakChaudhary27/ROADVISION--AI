// Firebase Client SDK Integration for RoadVision AI
import { initializeApp, getApps, getApp } from "firebase/app";
import { 
  getAuth, 
  signInWithEmailAndPassword, 
  createUserWithEmailAndPassword, 
  signOut, 
  sendPasswordResetEmail, 
  onAuthStateChanged,
  GoogleAuthProvider,
  signInWithPopup,
  signInWithRedirect,
  getRedirectResult,
  updateProfile,
  setPersistence,
  browserLocalPersistence,
  browserSessionPersistence
} from "firebase/auth";

// Read Firebase configuration strictly from Vite environment variables (import.meta.env)
const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY || "",
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN || "",
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID || "",
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET || "",
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID || "",
  appId: import.meta.env.VITE_FIREBASE_APP_ID || "",
  measurementId: import.meta.env.VITE_FIREBASE_MEASUREMENT_ID || ""
};

// Required environment variable specification
const REQUIRED_CONFIG = [
  { envKey: "VITE_FIREBASE_API_KEY", prop: "apiKey" },
  { envKey: "VITE_FIREBASE_AUTH_DOMAIN", prop: "authDomain" },
  { envKey: "VITE_FIREBASE_PROJECT_ID", prop: "projectId" },
  { envKey: "VITE_FIREBASE_APP_ID", prop: "appId" }
];

// Identify any missing or unconfigured required variables
export const missingConfigKeys = REQUIRED_CONFIG
  .filter(({ prop }) => {
    const val = firebaseConfig[prop];
    return !val || typeof val !== "string" || val.trim() === "" || val.includes("your_");
  })
  .map(({ envKey }) => envKey);

// Reliable configuration detection status
export const isFirebaseConfigured = missingConfigKeys.length === 0;

let app = null;
let auth = null;
let googleProvider = null;
let firebaseInitError = null;

if (isFirebaseConfigured) {
  try {
    // Avoid duplicate initialization during development and hot-module replacement
    app = getApps().length > 0 ? getApp() : initializeApp(firebaseConfig);
    auth = getAuth(app);
    googleProvider = new GoogleAuthProvider();
    googleProvider.addScope("email");
    googleProvider.addScope("profile");
    googleProvider.setCustomParameters({ prompt: "select_account" });
  } catch (error) {
    firebaseInitError = error.message;
    console.error("Firebase initialization failed:", error);
  }
} else {
  firebaseInitError = `Missing required environment variable(s): ${missingConfigKeys.join(", ")}`;
}

export { 
  app, 
  auth, 
  googleProvider,
  firebaseConfig,
  firebaseInitError,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut,
  sendPasswordResetEmail,
  onAuthStateChanged,
  signInWithPopup,
  signInWithRedirect,
  getRedirectResult,
  updateProfile,
  setPersistence,
  browserLocalPersistence,
  browserSessionPersistence
};
