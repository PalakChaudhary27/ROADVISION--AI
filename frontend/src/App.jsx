import React, { useState, useEffect } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import LandingPage from './pages/LandingPage';
import LoginPage from './pages/LoginPage';
import SignupPage from './pages/SignupPage';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import OverviewPage from './pages/OverviewPage';
import NewInspectionPage from './pages/NewInspectionPage';
import DetectionResultsPage from './pages/DetectionResultsPage';
import AnalyticsPage from './pages/AnalyticsPage';
import InspectionHistoryPage from './pages/InspectionHistoryPage';
import ReportsPage from './pages/ReportsPage';
import { Loader2 } from 'lucide-react';

function MainApp() {
  const { currentUser, loading: authLoading } = useAuth();

  // Helper to parse current URL path to internal route
  const getRouteFromPath = () => {
    const path = window.location.pathname.toLowerCase().replace(/\/$/, "");
    if (path === "" || path === "/" || path === "/home") return "landing";
    if (path === "/login") return "login";
    if (path === "/signup") return "signup";
    if (path === "/dashboard" || path === "/overview") return "overview";
    if (path === "/new-inspection") return "new-inspection";
    if (path === "/results") return "results";
    if (path === "/analytics") return "analytics";
    if (path === "/history") return "history";
    if (path === "/reports") return "reports";
    return "landing";
  };

  const [currentRoute, setCurrentRoute] = useState(getRouteFromPath);
  const [currentResult, setCurrentResult] = useState(null);
  const [selectedReportInspection, setSelectedReportInspection] = useState(null);
  const [inspections, setInspections] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [isSaved, setIsSaved] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Sync state with browser URL
  const navigateTo = (route, historyMode = "push") => {
    const normalizedRoute = route === "dashboard" ? "overview" : route;
    setCurrentRoute(normalizedRoute);
    window.scrollTo({ top: 0, behavior: "instant" });

    if (historyMode !== false) {
      let path = "/";
      if (normalizedRoute === "landing") path = "/";
      else if (normalizedRoute === "login") path = "/login";
      else if (normalizedRoute === "signup") path = "/signup";
      else if (normalizedRoute === "overview") path = "/dashboard";
      else path = `/${normalizedRoute}`;

      if (historyMode === "replace") {
        window.history.replaceState(null, "", path);
      } else {
        window.history.pushState(null, "", path);
      }
    }
  };

  // Listen for browser back / forward buttons
  useEffect(() => {
    const handlePopState = () => {
      setCurrentRoute(getRouteFromPath());
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  // Protected route enforcement
  useEffect(() => {
    if (authLoading) return;

    const publicRoutes = ["landing", "login", "signup"];
    const isPublic = publicRoutes.includes(currentRoute);

    if (!currentUser && !isPublic) {
      // Unauthenticated user attempting to access protected route -> redirect to login
      navigateTo("login", "replace");
    } else if (currentUser && (currentRoute === "login" || currentRoute === "signup")) {
      // Authenticated user on login/signup -> redirect to dashboard
      navigateTo("overview", "replace");
    }
  }, [currentUser, currentRoute, authLoading]);

  // Initial data loading from backend
  const loadData = async () => {
    setIsRefreshing(true);
    try {
      // 1. Health check
      const healthRes = await fetch('/health').catch(() => null);
      if (healthRes && healthRes.ok) {
        setBackendOnline(true);
      } else {
        setBackendOnline(false);
      }

      // 2. Fetch inspections
      const inspRes = await fetch('/api/inspections');
      if (inspRes.ok) {
        const inspData = await inspRes.json();
        setInspections(inspData);

        // If no active result, set currentResult to the latest saved one
        if (!currentResult && inspData.length > 0) {
          const fullRes = await fetch(`/api/inspections/${inspData[0].id}`);
          if (fullRes.ok) {
            const fullData = await fullRes.json();
            setCurrentResult(fullData);
            setIsSaved(true);
          }
        }
      }

      // 3. Fetch analytics
      const anaRes = await fetch('/api/analytics');
      if (anaRes.ok) {
        const anaData = await anaRes.json();
        setAnalytics(anaData);
      }
    } catch (err) {
      console.warn("Could not fetch backend data:", err);
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  // Handle successful prediction from NewInspectionPage
  const handleInferenceSuccess = (prediction) => {
    setCurrentResult(prediction);
    setIsSaved(false);
    navigateTo('results');
  };

  // Save current inspection to backend database
  const handleSaveInspection = async (resultToSave) => {
    if (!resultToSave || isSaved || isSaving) return;
    setIsSaving(true);
    try {
      const resp = await fetch('/api/inspections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(resultToSave),
      });

      if (!resp.ok) {
        throw new Error("Failed to save inspection to database.");
      }

      const resJson = await resp.json();
      setCurrentResult(resJson.inspection);
      setIsSaved(true);
      loadData();
    } catch (err) {
      alert(`Error saving inspection: ${err.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  // Open a saved inspection into Detection Results
  const handleViewInspection = async (id) => {
    try {
      const resp = await fetch(`/api/inspections/${id}`);
      if (!resp.ok) throw new Error("Could not retrieve inspection.");
      const data = await resp.json();
      setCurrentResult(data);
      setIsSaved(true);
      navigateTo('results');
    } catch (err) {
      alert(`Error loading inspection: ${err.message}`);
    }
  };

  // Delete an inspection
  const handleDeleteInspection = async (id) => {
    try {
      const resp = await fetch(`/api/inspections/${id}`, { method: 'DELETE' });
      if (!resp.ok) throw new Error("Could not delete inspection.");
      if (currentResult?.id === id) {
        setCurrentResult(null);
      }
      loadData();
    } catch (err) {
      alert(`Error deleting record: ${err.message}`);
    }
  };

  // Generate Report action
  const handleGenerateReport = async (item) => {
    if (item.images?.overlay) {
      setSelectedReportInspection(item);
    } else {
      const resp = await fetch(`/api/inspections/${item.id}`);
      if (resp.ok) {
        setSelectedReportInspection(await resp.json());
      }
    }
    navigateTo('reports');
  };

  // Loading state during auth initialization
  if (authLoading) {
    return (
      <div className="min-h-screen bg-[#0a0a0b] flex flex-col items-center justify-center text-white">
        <Loader2 className="w-8 h-8 animate-spin text-[#f5a524] mb-3" />
        <span className="font-mono text-xs text-white/60 tracking-wider uppercase">
          Initializing RoadVision Platform...
        </span>
      </div>
    );
  }

  // =========================================================================
  // PUBLIC ROUTES
  // =========================================================================
  if (currentRoute === "landing") {
    return <LandingPage onNavigate={navigateTo} />;
  }

  if (currentRoute === "login") {
    return <LoginPage onNavigate={navigateTo} />;
  }

  if (currentRoute === "signup") {
    return <SignupPage onNavigate={navigateTo} />;
  }

  // =========================================================================
  // PROTECTED APPLICATION WORKSPACE
  // =========================================================================
  return (
    <div className="flex h-screen bg-[#F5F7FA] overflow-hidden font-sans antialiased text-[#172435]">
      {/* Left Navigation Sidebar */}
      <Sidebar
        activePage={currentRoute}
        setActivePage={(page) => navigateTo(page)}
        backendOnline={backendOnline}
        onNavigatePublic={navigateTo}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        {/* Header Bar */}
        <Header
          activePage={currentRoute}
          setActivePage={(page) => navigateTo(page)}
          onRefresh={loadData}
          isRefreshing={isRefreshing}
          backendOnline={backendOnline}
          onNavigatePublic={navigateTo}
        />

        {/* Dynamic Page Views */}
        <main className="flex-1 overflow-y-auto p-6 md:p-8">
          {currentRoute === 'overview' && (
            <OverviewPage
              analytics={analytics}
              inspections={inspections}
              setActivePage={(page) => navigateTo(page)}
              onViewInspection={handleViewInspection}
            />
          )}

          {currentRoute === 'new-inspection' && (
            <NewInspectionPage
              onInferenceSuccess={handleInferenceSuccess}
              backendOnline={backendOnline}
            />
          )}

          {currentRoute === 'results' && (
            <DetectionResultsPage
              currentResult={currentResult}
              onSaveInspection={handleSaveInspection}
              onGenerateReport={handleGenerateReport}
              onNewInspection={() => navigateTo('new-inspection')}
              isSaved={isSaved}
              isSaving={isSaving}
            />
          )}

          {currentRoute === 'analytics' && (
            <AnalyticsPage
              analytics={analytics}
              inspections={inspections}
            />
          )}

          {currentRoute === 'history' && (
            <InspectionHistoryPage
              inspections={inspections}
              onViewInspection={handleViewInspection}
              onGenerateReport={handleGenerateReport}
              onDeleteInspection={handleDeleteInspection}
            />
          )}

          {currentRoute === 'reports' && (
            <ReportsPage
              inspections={inspections}
              selectedInspection={selectedReportInspection || currentResult}
              onSelectInspection={handleViewInspection}
            />
          )}
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <MainApp />
    </AuthProvider>
  );
}
