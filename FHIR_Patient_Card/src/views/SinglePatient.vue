<template>
  <main class="l-app__main" role="main">

    <header class="c-toolbar">
      <div class="c-cell">
        <div class="c-cell__media">
          <i class="icon-footprint"></i>
        </div>
        <div class="c-cell__content">
          <h1 class="c-toolbar__title">Patient {{ get(['id'], selectedPatient)}}</h1>
          <ol class="c-breadcrumb">
            <li class="c-breadcrumb__item">
              <router-link :to="{ name: 'patients' }" class="c-breadcrumb__link">Patients</router-link>
            </li>
            <li class="c-breadcrumb__item">Patient {{ get(['id'], selectedPatient)}}</li>
          </ol>
        </div>
      </div>
    </header>

    <table class="table table--data">
      <thead>
        <tr>
          <th>Key</th>
          <th>Value</th>
        </tr>
      </thead>
      <tfoot>
        <tr>
          <th>Key</th>
          <th>Value</th>
        </tr>
      </tfoot>
      <tbody>
        <tr v-if="get(['id'], selectedPatient)">
          <td data-label="Key">ID</td>
          <td data-label="Value">{{ get(['id'], selectedPatient) }}</td>
        </tr>

        <tr v-if="get(['name', 0], selectedPatient)">
          <td data-label="Key">Name</td>
          <td data-label="Value">
            {{ get(['name', 0, 'given', 0], selectedPatient) }}
            {{ get(['name', 0, 'given', 1], selectedPatient) }}
            <b>{{ get(['name', 0, 'family'], selectedPatient) }}</b>
          </td>
        </tr>

        <tr v-if="get(['meta', 'versionId'], selectedPatient)">
          <td data-label="Key">Version ID</td>
          <td data-label="Value">{{ get(['meta', 'versionId'], selectedPatient) }}</td>
        </tr>

        <tr v-if="get(['meta', 'lastUpdated'], selectedPatient)">
          <td data-label="Key">Last updated</td>
          <td data-label="Value">{{ new Date(get(['meta', 'lastUpdated'], selectedPatient)).toLocaleString() }}</td>
        </tr>

        <tr v-if="get(['address', 0, 'text'], selectedPatient)">
          <td data-label="Key">Address</td>
          <td data-label="Value">{{ get(['address', 0, 'text'], selectedPatient) }}</td>
        </tr>

        <tr v-if="get(['active'], selectedPatient)">
          <td data-label="Key">Active</td>
          <td data-label="Value">{{ get(['active'], selectedPatient) }}</td>
        </tr>

        <tr v-if="get(['birthDate'], selectedPatient)">
          <td data-label="Key">Birth date</td>
          <td data-label="Value">{{ get(['birthDate'], selectedPatient) }}</td>
        </tr>

        <tr v-if="get(['gender'], selectedPatient)">
          <td data-label="Key">Gender</td>
          <td data-label="Value">{{ get(['gender'], selectedPatient) }}</td>
        </tr>

        <tr v-if="get(['telecom', 0, 'system'], selectedPatient)">
          <td data-label="Key">{{ get(['telecom', 0, 'system'], selectedPatient).toUpperCase(1) }}</td>
          <td data-label="Value">{{ get(['telecom', 0, 'value'], selectedPatient) }}</td>
        </tr>

        <tr v-if="get(['telecom', 1, 'system'], selectedPatient)">
          <td data-label="Key">{{ get(['telecom', 1, 'system'], selectedPatient).toUpperCase() }}</td>
          <td data-label="Value">{{ get(['telecom', 1, 'value'], selectedPatient) }}</td>
        </tr>

        <tr v-if="get(['address', 0, 'city'], selectedPatient)">
          <td data-label="Key">City</td>
          <td data-label="Value">{{ get(['address', 0, 'city'], selectedPatient) }}</td>
        </tr>

        <tr v-if="get(['address', 0, 'country'], selectedPatient)">
          <td data-label="Key">Country</td>
          <td data-label="Value">{{ get(['address', 0, 'country'], selectedPatient) }}</td>
        </tr>

        <tr v-if="get(['address', 0, 'line', 0], selectedPatient)">
          <td data-label="Key">Address</td>
          <td data-label="Value">{{ get(['address', 0, 'line', 0], selectedPatient) }}</td>
        </tr>

        <tr v-if="get(['address', 0, 'postalCode'], selectedPatient)">
          <td data-label="Key">Postal code</td>
          <td data-label="Value">{{ get(['address', 0, 'postalCode'], selectedPatient) }}</td>
        </tr>

        <tr v-if="get(['address', 0, 'state'], selectedPatient)">
          <td data-label="Key">State</td>
          <td data-label="Value">{{ get(['address', 0, 'state'], selectedPatient) }}</td>
        </tr>
      </tbody>
    </table>
  </main>
</template>

<script>
  import { createNamespacedHelpers } from 'vuex'
  const { mapGetters } = createNamespacedHelpers('patient')

  export default {
    name: 'SinglePatientView',
    computed: mapGetters(['selectedPatient']),
    mounted () {
      this.patientID = (this.$route.params.patientID).replace(/Patient\//g, '')
      this.$store.dispatch('patient/getSinglePatient', this.patientID)
      .then(data => {
        mapGetters(['selectedPatient'])
      })
    },
    methods: {
      get (p, o) {
        return p.reduce((xs, x) => (xs && xs[x]) ? xs[x] : null, o)
      }
    }
  }
</script>
