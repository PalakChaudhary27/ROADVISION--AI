import React, { createContext, useContext, useState, useEffect } from "react";
import { 
  auth, 
  googleProvider,
  isFirebaseConfigured,
  missingConfigKeys,
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
} from "../services/firebase";

const AuthContext = createContext(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};

// Map Firebase error codes to clear, actionable messages
export const getAuthErrorMessage = (error) => {
  if (!error) return "";
  const code = error.code || "";
  
  switch (code) {
    case "auth/invalid-email":
      return "The email address is invalid or poorly formatted.";
    case "auth/user-disabled":
      return "This inspection account has been disabled. Contact your administrator.";
    case "auth/user-not-found":
    case "auth/wrong-password":
    case "auth/invalid-credential":
      return "Invalid email or password. Please verify your credentials.";
    case "auth/email-already-in-use":
      return "An account with this email address already exists. Please sign in.";
    case "auth/weak-password":
      return "Password should be at least 6 characters long.";
    case "auth/popup-closed-by-user":
    case "auth/cancelled-popup-request":
      return "Google sign-in window was closed before completing authentication.";
    case "auth/popup-blocked":
      return "Sign-in popup was blocked by your browser. Falling back to redirect authentication...";
    case "auth/operation-not-allowed":
      return "This sign-in method is not enabled in the Firebase Console. Go to Authentication > Sign-in method and enable Email/Password or Google.";
    case "auth/unauthorized-domain":
      return `Domain "${window.location.hostname}" is not authorized in Firebase Console. Add it under Authentication > Settings > Authorized domains.`;
    case "auth/api-key-not-valid":
    case "auth/invalid-api-key":
      return "Invalid Firebase API key. Check VITE_FIREBASE_API_KEY in frontend/.env.";
    case "auth/configuration-not-found":
      return "Firebase Authentication configuration not found for this project. Ensure Authentication is initialized in the Firebase Console.";
    case "auth/network-request-failed":
      return "Network connection error. Check your internet connection and retry.";
    case "auth/too-many-requests":
      return "Too many failed attempts. Access temporarily restricted. Try again later.";
    default:
      return error.message || "An authentication error occurred. Please try again.";
  }
};

const formatFirebaseUser = (user) => ({
  uid: user.uid,
  email: user.email,
  displayName: user.displayName || (user.email ? user.email.split("@")[0] : "Inspector"),
  photoURL: user.photoURL,
  emailVerified: user.emailVerified
});

export const AuthProvider = ({ children }) => {
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [authError, setAuthError] = useState(null);

  useEffect(() => {
    // Purge any legacy local mock key to guarantee no fake sessions persist
    try {
      localStorage.removeItem("roadvision_auth_user");
    } catch (_) {}

    if (!isFirebaseConfigured || !auth) {
      setCurrentUser(null);
      setLoading(false);
      return;
    }

    let isMounted = true;

    // Handle redirect result if returning from signInWithRedirect fallback
    getRedirectResult(auth)
      .then((result) => {
        if (isMounted && result?.user) {
          setCurrentUser(formatFirebaseUser(result.user));
        }
      })
      .catch((err) => {
        if (isMounted && err?.code) {
          setAuthError(getAuthErrorMessage(err));
        }
      });

    // Listen to Firebase auth state
    const unsubscribe = onAuthStateChanged(
      auth,
      (user) => {
        if (!isMounted) return;
        if (user) {
          setCurrentUser(formatFirebaseUser(user));
        } else {
          setCurrentUser(null);
        }
        setLoading(false);
      },
      (error) => {
        if (!isMounted) return;
        console.error("Firebase auth state error:", error);
        setAuthError(getAuthErrorMessage(error));
        setLoading(false);
      }
    );

    return () => {
      isMounted = false;
      unsubscribe();
    };
  }, []);

  const ensureFirebaseReady = () => {
    if (!isFirebaseConfigured || !auth) {
      const missing = missingConfigKeys && missingConfigKeys.length > 0 
        ? missingConfigKeys.join(", ") 
        : "VITE_FIREBASE_API_KEY, VITE_FIREBASE_AUTH_DOMAIN, VITE_FIREBASE_PROJECT_ID, VITE_FIREBASE_APP_ID";
      const msg = firebaseInitError || `Firebase is not configured. Missing required variable(s): ${missing}. Set them in frontend/.env and restart Vite.`;
      setAuthError(msg);
      throw new Error(msg);
    }
  };

  // Email & Password Sign In
  const login = async (email, password, rememberMe = true) => {
    setAuthError(null);
    ensureFirebaseReady();

    try {
      await setPersistence(
        auth,
        rememberMe ? browserLocalPersistence : browserSessionPersistence
      );
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      const userData = formatFirebaseUser(userCredential.user);
      setCurrentUser(userData);
      return userData;
    } catch (err) {
      const readable = getAuthErrorMessage(err);
      setAuthError(readable);
      throw new Error(readable);
    }
  };

  // Email & Password Sign Up
  const signup = async (email, password, displayName) => {
    setAuthError(null);
    ensureFirebaseReady();

    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      const user = userCredential.user;
      if (displayName) {
        await updateProfile(user, { displayName });
      }
      const userData = formatFirebaseUser(user);
      setCurrentUser(userData);
      return userData;
    } catch (err) {
      const readable = getAuthErrorMessage(err);
      setAuthError(readable);
      throw new Error(readable);
    }
  };

  // Google Sign In (Popup primary, Redirect fallback if popup blocked)
  const loginWithGoogle = async () => {
    setAuthError(null);
    ensureFirebaseReady();

    if (!googleProvider) {
      const msg = "GoogleAuthProvider is not initialized.";
      setAuthError(msg);
      throw new Error(msg);
    }

    try {
      const result = await signInWithPopup(auth, googleProvider);
      const userData = formatFirebaseUser(result.user);
      setCurrentUser(userData);
      return userData;
    } catch (err) {
      if (
        err.code === "auth/popup-blocked" ||
        err.code === "auth/operation-not-supported-in-this-environment"
      ) {
        // Fallback to Firebase redirect-based sign-in
        try {
          await signInWithRedirect(auth, googleProvider);
          return null;
        } catch (redirectErr) {
          const readable = getAuthErrorMessage(redirectErr);
          setAuthError(readable);
          throw new Error(readable);
        }
      }

      const readable = getAuthErrorMessage(err);
      setAuthError(readable);
      throw new Error(readable);
    }
  };

  // Password Reset
  const resetPassword = async (email) => {
    setAuthError(null);
    ensureFirebaseReady();

    try {
      await sendPasswordResetEmail(auth, email);
    } catch (err) {
      const readable = getAuthErrorMessage(err);
      setAuthError(readable);
      throw new Error(readable);
    }
  };

  // Sign Out
  const logout = async () => {
    setAuthError(null);
    if (auth) {
      try {
        await signOut(auth);
      } catch (err) {
        console.error("Signout error:", err);
      }
    }
    setCurrentUser(null);
  };

  const value = {
    currentUser,
    loading,
    authError,
    setAuthError,
    login,
    signup,
    loginWithGoogle,
    resetPassword,
    logout,
    isFirebaseConfigured,
    missingConfigKeys
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
};
