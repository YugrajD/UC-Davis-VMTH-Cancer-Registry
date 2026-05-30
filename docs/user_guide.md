# UC Davis VMTH Cancer Registry — User Guide

---

## Preface

### Document Structure

This document is the user guide for the UC Davis VMTH Cancer Registry web application. It covers all features available to end users, organized by role and workflow. The guide begins with an overview of the application and authentication, then walks through each tab and its functionality, followed by a roles and permissions reference, a troubleshooting and FAQ section, a glossary of terms, and contact information.

### Intended Audience

This guide is intended for veterinary researchers, clinicians, data reviewers, and administrative staff at the UC Davis Veterinary Medical Teaching Hospital (VMTH) and affiliated institutions. No technical background is required to use the application. Users with elevated roles (Reviewer, Admin) will find additional sections relevant to their workflows.

### Motivation

Canine cancer data collected at the VMTH represents a valuable epidemiological resource that has historically been difficult to query and analyze at scale. The VMTH Cancer Registry was developed to centralize de-identified pathology reports, classify them using a veterinary-specific NLP model (PetBERT), and make the resulting data accessible through an interactive research dashboard. The platform allows researchers to explore cancer incidence patterns across California counties, compare findings against environmental and human health datasets, and manage the full data ingestion and review pipeline through a single interface.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Navigation Overview](#2-navigation-overview)
3. [Overview Tab](#3-overview-tab)
4. [Cancer Types Tab](#4-cancer-types-tab)
5. [Cancer by Age Tab](#5-cancer-by-age-tab)
6. [Breed Disparities Tab](#6-breed-disparities-tab)
7. [Analysis Tab](#7-analysis-tab)
8. [Data Upload Tab](#8-data-upload-tab)
9. [Review Queue Tab](#9-review-queue-tab)
10. [Diagnosis Review Tab](#10-diagnosis-review-tab)
    - [10.1 Status Filter](#101-status-filter)
    - [10.2 Search and Filter](#102-search-and-filter)
    - [10.3 Diagnosis Detail Panel](#103-diagnosis-detail-panel)
    - [10.4 Review Actions](#104-review-actions)
    - [10.5 Pagination](#105-pagination)
11. [User Management Tab](#11-user-management-tab)
12. [Roles and Permissions](#12-roles-and-permissions)
13. [Requesting Access](#13-requesting-access)
14. [Troubleshooting / FAQ](#14-troubleshooting--faq)
15. [Glossary](#15-glossary)
16. [Contact Information](#16-contact-information)

---

## 1. Getting Started

### 1.1 Signing In

The application uses Supabase Auth for single sign-on. Click **Sign In** in the top-right corner of the navigation bar. You can sign in with your Google account or with an email and password combination.

Without signing in, you can still browse the public dashboards (Overview, Cancer Types, Cancer by Age, Breed Disparities, Analysis). Signing in unlocks the ability to request expanded roles, submit data, and access role-specific tabs.

### 1.2 Your Account

After signing in, your email address appears in the top-right corner of the navigation bar. Click it to sign out. Hovering over your email shows your currently assigned roles (e.g., Uploader, Reviewer, Admin).

---

## 2. Navigation Overview

The top navigation bar contains tabs that appear based on your role. Tabs with pending items display a colored badge showing the pending count (e.g., **Diagnosis Review 12**).

| Tab | Who Can See It |
|-----|----------------|
| Overview | Everyone |
| Cancer Types | Everyone |
| Cancer by Age | Everyone |
| Breed Disparities | Everyone |
| Analysis | Everyone |
| Data Upload | Uploader or Admin |
| Review Queue | Reviewer or Admin |
| Diagnosis Review | Uploader, Reviewer, or Admin |
| User Management | Admin only |

---

## 3. Overview Tab

The Overview tab is the default landing page. It shows a summary of all cancer cases in the registry alongside an interactive California map.

### 3.1 Summary Cards

At the top of the page, four cards display high-level registry statistics:

- **Total Cases** — number of individual cancer diagnoses
- **Total Patients** — number of unique dogs in the registry
- **Counties Represented** — number of California counties with at least one case
- **Year Range** — earliest and latest diagnosis years in the current filtered view

### 3.2 Filter Panel

The right-side panel contains filters that apply across the entire tab simultaneously:

- **Sex** — filter by Intact Male, Neutered Male, Intact Female, or Spayed Female
- **Age Group** — filter by age bracket: Young (0–2 yrs), Juvenile (3–5 yrs), Adult (6–8 yrs), Old (9–11 yrs), or Senior (≥12 yrs)
- **Cancer Type** — select one or more cancer type categories (e.g., Hematopoietic, Integumentary)
- **Breed** — type to search for a specific breed
- **Year Range** — drag the slider to restrict cases to a date range
- **Rate Type** — toggle between raw case count and incidence rate (cases per 10,000 dogs)

All filters update the map, summary cards, and county table simultaneously.

### 3.3 Choropleth Map

The map shows California counties shaded by the currently selected rate type. Darker shading indicates higher case counts or incidence rates.

- **Hover** over a county to highlight it and see a tooltip with the county name and value.
- **Click** a county to filter the table to that county. Click the same county again to deselect.

The map defaults to a fitted California view. Use the **Reset View** button above the map to return to the default zoom at any time.

### 3.4 County Table

Below the map, a hierarchical table shows data organized as follows:

1. **California** — statewide total
2. **UC Davis Catchment Area** — 16 counties in the primary VMTH service region
3. **Other Counties** — remaining California counties with data
4. **Individual counties** — listed under each group

Each row shows case count, patient count, and incidence rate. Clicking any county row highlights it on the map.

---

## 4. Cancer Types Tab

This tab breaks down cases by cancer classification according to the **Vet-ICD-O-Canine-1** veterinary oncology taxonomy. Non-Cancer predictions are excluded from this view.

### 4.1 Category Filter Pills

A row of pill buttons at the top of the tab allows filtering by broad cancer category (e.g., Hematopoietic, Musculoskeletal, Integumentary). Each pill displays the case count for that category. Click a pill to isolate that category; click it again to deselect.

### 4.2 Bar Chart

A horizontal bar chart shows the top 10 most common cancer types within the current filter selection. Bars are labeled with the cancer type name and case count. The chart updates in real time as you apply category filters or adjust the global filters from the filter panel.

---

## 5. Cancer by Age Tab

This tab explores how cancer type patterns differ across dog age groups. It is public and requires no sign-in.

### 5.1 Age Group Selector

A dropdown at the top of the tab lets you select one of five age brackets:

| Label | Age Range |
|-------|-----------|
| Young | 0–2 years |
| Juvenile | 3–5 years |
| Adult | 6–8 years |
| Old | 9–11 years |
| Senior | ≥ 12 years |

Select an age group to load data for that bracket. No data is shown until a group is selected.

### 5.2 Summary Stats

Once an age group is selected, a summary line shows:

- Total case diagnoses for that age group
- Number of counties with at least one case
- Sex distribution (counts per sex category)

### 5.3 Cancer Type Breakdown

A horizontal bar chart displays the top 15 most common cancer types for the selected age group, with case counts. Bars are sized relative to the most common type.

### 5.4 County Map

A California choropleth map shades counties by case count for the selected age group. Hover over a county to see its name and count.

---

## 6. Breed Disparities Tab

This tab allows exploration of cancer patterns for a specific dog breed. Non-Cancer predictions are excluded from all breed statistics.

### 6.1 Breed Search

Type at least 2 characters in the search box to see matching breeds from the registry. Select a breed to load its detail view. Frequently selected breeds may also appear as quick-select buttons.

### 6.2 Breed Detail View

Once a breed is selected, the detail view displays:

- **Case Count and Patient Count** for that breed
- **County Map** — a choropleth showing geographic concentration of cases for that breed
- **Cancer Type Frequency** — bar chart of the most common cancer types in that breed
- **Sex Distribution** — breakdown of intact vs. neutered males and females
- **Top Counties** — ranked list of counties by case count for that breed

All views respect the global filters (year range, sex, cancer type) from the filter panel.

---

## 7. Analysis Tab

The Analysis tab is an advanced multi-variable comparison tool for researchers investigating correlations between canine cancer and environmental or demographic factors.

### 7.1 Variable Selection

Two dropdown menus allow selection of an **X variable** and a **Y variable** from over 50 options organized into groups:

- **VMTH Cancer Data** — case counts by county
- **CDPR Pesticide Data** — total pesticide use, 2015 snapshot, 2019 snapshot, 10-year change
- **EPA Superfund** — number of Superfund sites per county
- **CalEnviroScreen** — 24 environmental justice indicators including pollution burden, health outcomes, and socioeconomic factors (e.g., Ozone, PM2.5, Pesticides, Poverty Rate, Education, Cardiovascular Disease)
- **CCR Human Cancer** — human cancer incidence rates from the California Cancer Registry, broken out by cancer site

### 7.2 Scatter Plot

A scatter plot shows each California county as a point, with the X variable on the horizontal axis and the Y variable on the vertical axis. Hover over a point to see the county name and exact values. This view helps identify correlations or outliers across the county-level dataset.

### 7.3 Four-Map Grid

Below the scatter plot, four side-by-side maps display:

- **Top-left** — VMTH cancer case counts (baseline reference)
- **Other three** — the selected X and Y variables as color-encoded county layers

Maps support pan and zoom. Click **Reset View** to return all maps to the California bounding box.

### 7.4 Yearly Trends Chart

At the bottom of the tab, a line chart shows VMTH case counts by year, drawn from real registry data. If a pesticide variable is selected as X or Y, a second line overlays pesticide trend data for visual temporal comparison.

---

## 8. Data Upload Tab

The Data Upload tab allows users with the **Uploader** role to submit new pathology report datasets for classification and ingestion into the registry.

### 8.1 Uploading a File

1. Drag and drop a CSV or XLSX file onto the upload zone, or click the zone to open a file picker.
2. A preview modal appears showing the file name, number of rows, number of columns, and file size. Required columns are shown with human-readable display names to make it easy to confirm the file has the expected structure.
3. Review the preview to confirm the correct file is selected.
4. Click **Submit** to queue the file for review and ingestion.

Accepted formats: `.csv`, `.xlsx`. Maximum file size is 50 MB. XLSX files are automatically converted to CSV during upload.

### 8.2 Upload Status List

Below the upload zone, a list shows all files you have previously submitted. Each entry displays:

- **Filename** and upload date
- **Status badge**: `pending_review` (yellow), `processing` (blue), `completed` (green), `failed` (red), `rejected` (gray)
- **Pipeline stage indicator** for jobs that are actively processing (see Section 9 for stage details)

The list refreshes automatically every 30 seconds while any job is in a pending or processing state.

### 8.3 Requesting a Data Export

Authenticated users can request a CSV export of the current registry data by clicking **Request Export**. You will be prompted to provide an optional reason for the request. An admin reviews the request and, if approved, generates a one-time download link delivered to your email.

### 8.4 Requesting the Uploader Role

If the Data Upload tab is not visible, you do not yet have the Uploader role. See Section 13 for instructions on requesting access.

---

## 9. Review Queue Tab

The Review Queue tab is available to users with the **Reviewer** or **Admin** role. It shows all ingestion jobs and allows reviewers to approve or reject uploaded files before they are processed by the PetBERT classification pipeline.

### 9.1 Job List

Each job card displays:

- **Filename**, file size, row count, and submission date
- **Submitter email**
- **Status badge** (pending_review, processing, completed, failed, rejected)
- **Pipeline stage indicator** showing progress through the four stages:
  1. Upload Validation
  2. GCP Batch Classification (PetBERT)
  3. Database Ingestion
  4. Complete

The active stage shows a spinner; completed stages show a checkmark.

### 9.2 Reviewing a Job

For jobs with status `pending_review`, the following actions are available:

- **Preview** — opens a modal showing the first 20 rows of the uploaded CSV. Use this to verify data format and content before approving.
- **Approve** — sends the file through the PetBERT classification pipeline. The job status changes to `processing`.
- **Reject** — rejects the submission with an optional rejection reason. The submitter sees the rejection status in their upload list.
- **Cancel** — withdraws a job that has not yet been approved. Only available to the original submitter or an admin.

### 9.3 Archive Filter

By default, the queue shows active jobs (status: pending_review and processing). Toggle **Show Archived** to also display completed, failed, and rejected jobs.

---

## 10. Diagnosis Review Tab

After a file is classified by PetBERT, each individual diagnosis prediction appears in the Diagnosis Review tab for human verification. This tab is available to users with the **Reviewer** or **Admin** role for triage, and to **Uploader** or **Admin** users for auditing their own submitted data.

### 10.1 Status Filter

A row of filter pills at the top of the page controls which diagnoses appear in the queue:

| Filter | Who can see it | What it shows |
|--------|---------------|---------------|
| **Pending** | Reviewer, Admin | Diagnoses not yet reviewed — the primary triage queue |
| **Confirmed** | Uploader, Admin | Diagnoses where a reviewer accepted the PetBERT prediction |
| **Corrected** | Uploader, Admin | Diagnoses where a reviewer assigned a different cancer type |
| **Rejected** | Uploader, Admin | Diagnoses a reviewer marked as invalid |
| **All** | Uploader, Admin | All diagnoses regardless of status |

Uploaders and admins use the non-Pending filters to audit the results of their own submitted jobs. Non-admin uploaders see only diagnoses from jobs they submitted; admins see all.

### 10.2 Search and Filter

Below the status filter pills, a row of inputs lets you narrow the queue before paging through it.

| Filter | Available to | Behavior |
|--------|-------------|----------|
| **Year** | Uploader, Reviewer, Admin | Type a 4-digit year to show only diagnoses for patients whose diagnosis date falls in that year. Results update automatically 400 ms after you stop typing. If the year entered is outside the available data range (1900–2100) a warning appears below the field and no year filter is applied. Clearing the field removes the filter. |
| **Patient ID** | Uploader, Reviewer, Admin | Type part of a patient identifier (anonymized ID) to search by substring. The queue updates automatically a short pause after each keystroke. If no patient matches, the queue shows a "No patient found matching…" message. |
| **Clinic** | Admin only | A dropdown listing every uploader email that has submitted at least one job. Select a clinic to see only diagnoses from that submitter's jobs. Defaults to "All clinics." |
| **Type** | Uploader, Reviewer, Admin | A dropdown positioned after Clinic. Filter by predicted cancer type name (substring match). |
| **Cancer Group** | Uploader, Reviewer, Admin | Filter by prediction category: **All** (default), **Cancer**, **Non-Cancer**, or **Unidentified**. Use this to focus on confirmed cancer predictions or to audit Non-Cancer and ambiguous results separately. |

All filters combine with the active status pill. Changing any filter resets the queue to page 1.

### 10.3 Diagnosis Detail Panel

Selecting a diagnosis from the queue opens a detail panel on the right with:

- **Patient identifier** (anonymized) and diagnosis index
- **PetBERT Prediction** — predicted cancer type, Vet-ICD-O code, matched label term, and confidence score
- **Confidence Bar** — color-coded indicator (red below 50%, amber 50–80%, green 80% and above)
- **Top-1 vs Top-2 margin** — gap between the model's top-ranked and second-ranked prediction; a small margin signals ambiguity
- **Prediction method** — how the classification was produced (e.g., `embedding`, `low_confidence`)
- **Pathology report text** — the raw text from the uploaded file that PetBERT classified, with a **Show more / Show less** toggle for long reports
- **Original PetBERT prediction** — shown when a reviewer has since corrected the prediction, so you can see what the model originally said
- **Review history** — every prior action with timestamps and reviewer email

### 10.4 Review Actions

Review actions are available for diagnoses in **Pending**, **Confirmed**, and **Corrected** status. Rejected diagnoses are read-only.

- **Confirm** — mark the PetBERT prediction as correct. The diagnosis enters the public dashboard statistics.
- **Correct** — type a different cancer type name and optionally a corrected ICD-O code. The corrected values are saved and attributed to your account. If the cancer type is new (not in the existing taxonomy), it is created as unconfirmed and queued for admin sign-off.
- **Reject** — mark the prediction as not a valid cancer diagnosis. Rejected diagnoses are excluded from dashboard statistics.

All actions are recorded in the review history with your identity and timestamp.

### 10.5 Pagination

Use the **Prev** and **Next** buttons at the top of the queue to move between pages (50 diagnoses per page). A page number input lets you jump directly to any page; the total page count is displayed alongside it. The badge in the navigation bar shows the current pending count.

---

## 11. User Management Tab

The User Management tab is available to **Admin** users only.

### 11.1 Role Assignment

To update roles for any user:

1. Enter the user's email address in the search box.
2. Toggle the checkboxes for **Admin**, **Reviewer**, and/or **Uploader** as needed.
3. Click **Save** to apply changes immediately.

Role changes take effect on the user's next page load or sign-in.

> **Note:** Admins cannot edit their own roles or the roles of other admins. The form is read-only when viewing your own account or another admin's account. This prevents accidental lockout.

### 11.2 Pending Role Requests

Users who request a role through the application appear in the **Role Requests** queue. Each entry shows the user's email, the requested role, an optional reason, and the request date. Click **Approve** to grant the role or **Deny** to reject the request. The user's tab visibility updates automatically after approval.

### 11.3 Pending Export Requests

Users who request a data export appear in the **Export Requests** queue. Each entry shows the user's email, their stated reason, and the request date. Click **Approve** to generate a one-time download link (delivered to the user's email) or **Deny** to reject.

### 11.4 Refresh Materialized Views

After a large ingestion completes, dashboard statistics may be stale because the application caches aggregated data. Click **Refresh Views** to rebuild the materialized views in the database. This operation takes a few seconds and is safe to run at any time.

---

## 12. Roles and Permissions

| Action | Anonymous | Authenticated | Uploader | Reviewer | Admin |
|--------|:---------:|:-------------:|:--------:|:--------:|:-----:|
| View Overview, Cancer Types, Cancer by Age, Breed Disparities, Analysis | Yes | Yes | Yes | Yes | Yes |
| Sign in / sign out | — | Yes | Yes | Yes | Yes |
| Request a role | — | Yes | Yes | Yes | Yes |
| Request a data export | — | Yes | Yes | Yes | Yes |
| Upload a CSV/XLSX file | — | — | Yes | Yes | Yes |
| Browse diagnoses by status (audit view, own jobs only) | — | — | Yes | — | Yes |
| Approve/reject ingestion jobs | — | — | — | Yes | Yes |
| Review and correct diagnoses | — | — | — | Yes | Yes |
| Assign user roles | — | — | — | — | Yes |
| Approve role and export requests | — | — | — | — | Yes |
| Refresh materialized views | — | — | — | — | Yes |

---

## 13. Requesting Access

### 13.1 Requesting a Role

1. Sign in to the application.
2. Navigate to the **Data Upload** tab. If it is not visible, a **Request Access** link appears in the navigation bar.
3. Click **Request Role**, select the role you need (Uploader or Reviewer), and optionally explain why.
4. An admin will review your request. You will see your request listed as pending until it is resolved.
5. Once approved, the relevant tab will appear in your navigation on your next page load.

---

## 14. Troubleshooting / FAQ

### 14.1 Basic Troubleshooting

If you experience issues accessing or using the application, follow these steps in order:

1. **Check your internet connection.** The application requires an active internet connection. Verify connectivity before attempting other steps.
2. **Refresh the page.** Some display issues resolve after a page reload.
3. **Clear your browser cache.** Cached pages can cause stale UI state. Clear your cache and reload if refreshing alone does not help.
4. **Try an incognito or private window.** Authentication tokens and cookies are refreshed in private browsing mode, which can resolve sign-in issues.
5. **Contact support.** If none of the above steps resolve the issue, see Section 16 for contact information.

### 14.2 Frequently Asked Questions

1. **The Data Upload tab is not visible. What should I do?**
   You do not yet have the Uploader role. Follow the steps in Section 13.1 to request access. Once an admin approves your request, the tab will appear on your next page load.

2. **I uploaded a file but the status has not changed for a long time. Is something wrong?**
   The status list refreshes automatically every 30 seconds. If a job has been in `pending_review` for an extended period, it is waiting for a reviewer to approve or reject it. If a job is stuck in `processing`, contact a reviewer or admin to investigate.

3. **Can I upload an Excel file, or does it need to be a CSV?**
   Both `.csv` and `.xlsx` formats are accepted. XLSX files are automatically converted to CSV during upload. The maximum file size for either format is 50 MB.

4. **A PetBERT prediction looks incorrect. What should I do?**
   Navigate to the Diagnosis Review tab, locate the diagnosis in question, and use the **Correct** action to select the appropriate cancer type from the Vet-ICD-O-Canine-1 dropdown. If the prediction is entirely invalid (not a cancer diagnosis), use **Reject** to exclude it from the dashboard statistics.

5. **Why does the dashboard not reflect data from a recently completed ingestion?**
   The dashboard uses cached aggregated statistics (materialized views) that must be refreshed after a large ingestion. If you are an admin, click **Refresh Views** in the User Management tab. If you are not an admin, contact one to request a refresh.

6. **I approved an export request, but the user says they did not receive the download link. What should I do?**
   The download link is sent to the user's registered email address. Ask the user to check their spam folder. If the email is not found, you may need to re-approve the request to generate a new link.

7. **The map is not displaying correctly. What should I try?**
   Try resetting the map view using the **Reset View** button. If the map remains blank, clear your browser cache and reload the page. If the issue persists, contact support.

8. **Can I export data without going through the export request process?**
   No. All data exports require admin approval to ensure appropriate data governance. Submit an export request through the Data Upload tab and provide a reason; an admin will review it promptly.

9. **I see a "Too many requests" message in Diagnosis Review. What does this mean?**
   The backend limits each authenticated user to 120 requests per minute on the Diagnosis Review endpoints. This can be reached if you page through results very quickly or if you were previously typing search terms rapidly before the debounce behavior was applied. When it occurs, the application shows "Too many requests — please try again in a moment." Wait a few seconds and the limit resets automatically. No sign-out is required.

10. **The Cancer by Age tab shows no data after I select an age group. Why?**
    Age group data depends on birth date information being present in the ingested records. If the predictions CSV used during ingestion did not include birth date data, age group filters will return empty results. Contact an admin to confirm whether the current dataset includes age information.

---

## 15. Glossary

| Term | Definition |
|------|------------|
| **PetBERT** | A BERT-based natural language processing model pretrained on veterinary electronic health records, used to classify pathology report text into cancer types. |
| **Vet-ICD-O-Canine-1** | The veterinary oncology classification standard used to categorize canine cancer diagnoses in this registry. |
| **Choropleth map** | A map in which geographic regions (counties) are shaded in proportion to a statistical variable, such as cancer case count or incidence rate. |
| **Incidence rate** | The number of new cancer cases per 10,000 dogs in a given county and time period. |
| **Materialized view** | A pre-computed database query result stored for fast retrieval. Must be refreshed after data changes to reflect the latest ingested records. |
| **CalEnviroScreen** | A statewide environmental justice screening tool developed by the California Office of Environmental Health Hazard Assessment (OEHHA) that scores communities by pollution burden and population vulnerability. |
| **CDPR** | California Department of Pesticide Regulation. Provides county-level pesticide use data used in the Analysis tab. |
| **CCR** | California Cancer Registry. Provides human cancer incidence rates used for comparison in the Analysis tab. |
| **Ingestion** | The process of validating, classifying, and loading uploaded pathology report data into the registry database. |
| **Age group** | One of five age brackets used to filter and segment cancer data: Young (0–2 yrs), Juvenile (3–5 yrs), Adult (6–8 yrs), Old (9–11 yrs), Senior (≥12 yrs). |
| **Non-Cancer** | A PetBERT prediction category indicating the report text does not describe a malignant diagnosis. Non-Cancer predictions are excluded from all public dashboard statistics and the Cancer Types and Breed Disparities tabs, but remain visible in the Diagnosis Review tab via the Cancer Group filter. |
| **Reciprocal best hit** | A classification approach where the strongest match between two items is confirmed in both directions. |
| **Uploader** | A user role that grants the ability to submit CSV or XLSX files for ingestion. |
| **Reviewer** | A user role that grants the ability to approve or reject ingestion jobs and review individual diagnosis predictions. |
| **Admin** | A user role with full access including user management, role and export request resolution, and materialized view refresh. |

---

## 16. Contact Information

For questions about access or registry operations, contact the VMTH Cancer Registry administrator at the UC Davis School of Veterinary Medicine.

For technical issues with the application, file a report via the project's GitHub repository.
