import React, { useState, useEffect } from "react";
import { 
  Scan, 
  ArrowRight, 
  Eye, 
  EyeOff, 
  Lock, 
  Mail, 
  AlertCircle, 
  CheckCircle2, 
  Loader2,
  AlertTriangle
} from "lucide-react";
import { useAuth } from "../context/AuthContext";

export default function LoginPage({ onNavigate }) {
  const { 
    login, 
    loginWithGoogle, 
    resetPassword, 
    authError, 
    setAuthError, 
    isFirebaseConfigured,
    missingConfigKeys 
  } = useAuth();
  
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(true);
  const [loading, setLoading] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);
  
  // Password reset state
  const [showForgotModal, setShowForgotModal] = useState(false);
  const [resetEmail, setResetEmail] = useState("");
  const [resetSent, setResetSent] = useState(false);
  const [resetLoading, setResetLoading] = useState(false);

  // Clear previous page errors when mounting LoginPage
  useEffect(() => {
    setAuthError(null);
  }, [setAuthError]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email || !password) {
      setAuthError("Please provide both email and password.");
      return;
    }

    try {
      setLoading(true);
      await login(email, password, rememberMe);
      onNavigate("dashboard");
    } catch (err) {
      // Error surfaced via authError
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    try {
      setGoogleLoading(true);
      const user = await loginWithGoogle();
      if (user) {
        onNavigate("dashboard");
      }
    } catch (err) {
      // Error surfaced via authError
    } finally {
      setGoogleLoading(false);
    }
  };

  const handlePasswordReset = async (e) => {
    e.preventDefault();
    if (!resetEmail) return;
    try {
      setResetLoading(true);
      await resetPassword(resetEmail);
      setResetSent(true);
    } catch (err) {
      // Error surfaced via authError
    } finally {
      setResetLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0b] text-[#f3f1ec] flex flex-col justify-between selection:bg-[#f5a524] selection:text-[#0a0a0b]">
      {/* Top Bar */}
      <header className="border-b border-white/10 px-6 py-4 flex items-center justify-between">
        <button 
          type="button"
          className="flex items-center gap-3 cursor-pointer text-left"
          onClick={() => onNavigate("landing")}
        >
          <div className="w-8 h-8 rounded-full bg-[#f5a524] flex items-center justify-center font-bold text-[#0a0a0b]">
            <Scan className="w-4 h-4" />
          </div>
          <span className="font-bold tracking-tight text-white">RoadVision<span className="text-[#f5a524]">.AI</span></span>
        </button>

        <button 
          type="button"
          onClick={() => onNavigate("landing")}
          className="text-xs font-mono text-white/60 hover:text-white"
        >
          ← Back to Overview
        </button>
      </header>

      {/* Main Authentication Card */}
      <main className="flex-1 flex items-center justify-center p-4 sm:p-6">
        <div className="w-full max-w-md bg-[#121214] border border-white/10 rounded-2xl p-6 sm:p-8 shadow-2xl relative">
          {/* Header */}
          <div className="mb-6">
            <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-white/5 border border-white/10 text-[#f5a524] text-[10px] font-mono tracking-widest uppercase mb-3">
              ENTERPRISE AUTHENTICATION
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-white mb-2">
              Welcome Back
            </h1>
            <p className="text-sm text-white/60">
              Sign in to access your intelligent pavement inspection workspace.
            </p>
          </div>

          {/* Firebase Configuration Status Notice */}
          {!isFirebaseConfigured && (
            <div className="mb-5 p-3.5 rounded-xl bg-amber-950/30 border border-[#f5a524]/30 text-[#f5a524] text-xs flex items-start gap-2.5">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <div className="leading-relaxed">
                <span className="font-semibold block mb-0.5">Firebase Environment Not Configured</span>
                <span className="text-white/70">
                  Missing required variable(s): <code className="font-mono text-[#f5a524]">{missingConfigKeys?.join(", ") || "VITE_FIREBASE_API_KEY"}</code>. Set in <code className="font-mono text-[#f5a524]">frontend/.env</code> and restart Vite.
                </span>
              </div>
            </div>
          )}

          {/* Error Message */}
          {authError && (
            <div className="mb-6 p-3.5 rounded-xl bg-red-950/40 border border-red-500/30 text-red-300 text-xs flex items-start gap-2.5">
              <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
              <span>{authError}</span>
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-mono text-white/70 mb-1.5 uppercase tracking-wider">
                Inspection Email
              </label>
              <div className="relative">
                <Mail className="w-4 h-4 text-white/40 absolute left-3.5 top-1/2 -translate-y-1/2" />
                <input 
                  type="email"
                  required
                  placeholder="engineer@roadauthority.gov"
                  value={email}
                  onChange={(e) => { setEmail(e.target.value); if (authError) setAuthError(null); }}
                  className="w-full h-11 pl-10 pr-3.5 rounded-xl bg-[#0a0a0b] border border-white/10 text-white placeholder-white/30 text-sm focus:outline-none focus:border-[#f5a524] transition-colors"
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-mono text-white/70 uppercase tracking-wider">
                  Password
                </label>
                <button 
                  type="button"
                  onClick={() => { setShowForgotModal(true); setAuthError(null); }}
                  className="text-xs text-[#f5a524] hover:underline"
                >
                  Forgot password?
                </button>
              </div>
              <div className="relative">
                <Lock className="w-4 h-4 text-white/40 absolute left-3.5 top-1/2 -translate-y-1/2" />
                <input 
                  type={showPassword ? "text" : "password"}
                  required
                  placeholder="••••••••••••"
                  value={password}
                  onChange={(e) => { setPassword(e.target.value); if (authError) setAuthError(null); }}
                  className="w-full h-11 pl-10 pr-10 rounded-xl bg-[#0a0a0b] border border-white/10 text-white placeholder-white/30 text-sm focus:outline-none focus:border-[#f5a524] transition-colors"
                />
                <button 
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-white/40 hover:text-white"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between pt-1">
              <label className="flex items-center gap-2 cursor-pointer text-xs text-white/70">
                <input 
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)}
                  className="w-3.5 h-3.5 rounded border-white/20 bg-[#0a0a0b] text-[#f5a524] focus:ring-0"
                />
                <span>Remember this terminal session</span>
              </label>
            </div>

            <button 
              type="submit"
              disabled={loading || googleLoading}
              className="btn-pivai-amber w-full justify-center h-11 mt-2 text-sm font-semibold disabled:opacity-60"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin mr-2" />
                  Authenticating...
                </>
              ) : (
                <>
                  Sign In
                  <ArrowRight className="w-4 h-4 ml-1" />
                </>
              )}
            </button>
          </form>

          {/* Divider */}
          <div className="relative my-6 text-center">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-white/10"></div>
            </div>
            <span className="relative px-3 bg-[#121214] text-[11px] font-mono text-white/40 uppercase">
              Or Continue With
            </span>
          </div>

          {/* Google Sign In */}
          <button 
            type="button"
            onClick={handleGoogleSignIn}
            disabled={loading || googleLoading}
            className="w-full h-11 rounded-full border border-white/15 bg-white/5 hover:bg-white/10 text-white text-xs font-semibold flex items-center justify-center gap-3 transition-colors disabled:opacity-60"
          >
            {googleLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <svg className="w-4 h-4" viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z" />
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z" />
              </svg>
            )}
            <span>{googleLoading ? "Connecting to Google..." : "Continue with Google"}</span>
          </button>

          {/* Footer Link */}
          <div className="mt-6 text-center text-xs text-white/50">
            Don't have an inspection account?{" "}
            <button 
              type="button"
              onClick={() => onNavigate("signup")}
              className="text-[#f5a524] hover:underline font-semibold"
            >
              Create Account
            </button>
          </div>
        </div>
      </main>

      {/* Forgot Password Modal */}
      {showForgotModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="w-full max-w-sm bg-[#121214] border border-white/15 rounded-2xl p-6 shadow-2xl">
            <h3 className="text-lg font-bold text-white mb-2">Reset Password</h3>
            <p className="text-xs text-white/60 mb-4">
              Enter your registered inspection email and we will dispatch a password recovery link.
            </p>

            {authError && (
              <div className="mb-4 p-3 rounded-xl bg-red-950/40 border border-red-500/30 text-red-300 text-xs flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                <span>{authError}</span>
              </div>
            )}

            {resetSent ? (
              <div className="p-3 bg-emerald-950/30 border border-emerald-500/30 rounded-xl text-emerald-300 text-xs flex items-center gap-2 mb-4">
                <CheckCircle2 className="w-4 h-4 shrink-0" />
                <span>Password reset link sent! Check your inbox.</span>
              </div>
            ) : (
              <form onSubmit={handlePasswordReset} className="space-y-4 mb-4">
                <input 
                  type="email"
                  required
                  placeholder="your-email@roadauthority.gov"
                  value={resetEmail}
                  onChange={(e) => setResetEmail(e.target.value)}
                  className="w-full h-10 px-3 rounded-xl bg-[#0a0a0b] border border-white/15 text-white text-xs focus:outline-none focus:border-[#f5a524]"
                />
                <button 
                  type="submit"
                  disabled={resetLoading}
                  className="btn-pivai-amber w-full justify-center h-10 text-xs font-semibold"
                >
                  {resetLoading ? "Sending..." : "Dispatch Reset Link"}
                </button>
              </form>
            )}

            <button 
              type="button"
              onClick={() => { setShowForgotModal(false); setResetSent(false); setAuthError(null); }}
              className="w-full text-center text-xs text-white/50 hover:text-white"
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Footer info */}
      <footer className="border-t border-white/10 px-6 py-4 text-center text-xs text-white/40 font-mono">
        RoadVision AI · Secured with Firebase Authentication
      </footer>
    </div>
  );
}
