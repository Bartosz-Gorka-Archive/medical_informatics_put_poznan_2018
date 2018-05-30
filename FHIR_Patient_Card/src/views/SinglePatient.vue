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
          <td data-label="Value">
            Current version {{ get(['meta', 'versionId'], selectedPatient) }}
            <template v-for="num in this.totalVersions - 1">
              <router-link :to="{ name: 'single-versioned-patient', params: { patientID: get(['id'], selectedPatient), versionNumber: num }}">
                [version {{ num }}]
              </router-link>
            </template>
          </td>
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

        <tr v-if="get(['telecom', 0, 'value'], selectedPatient)">
          <td data-label="Key">{{ get(['telecom', 0, 'system'], selectedPatient) }}</td>
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

    <div class="divider u-mt-20"><span>Record edition</span></div>

    <div class="u-mt-20 l-row">
      <div class="l-col-4@md"></div>
      <div class="l-col-4@md">
        <select class="form-select" v-model="gender">
          <option value="male">Male</option>
          <option value="female">Female</option>
        </select>

        <input type="date" class="form-input" placeholder="Birthdate" v-model="birthDate">

        <button class="btn btn--info u-mt-10" @click="sendUpdate()">Update record</button>
      </div>
      <div class="l-col-4@md"></div>
    </div>

    <div class="divider u-mt-20"><span>Timeline</span></div>

    <div class="u-mt-20 l-row">
      <div class="l-col-4@md"></div>
      <div class="l-col-4@md">
        <input type="date" class="form-input" placeholder="Birthdate" v-model="selectedDate">

        <button class="btn btn--default u-mt-10" @click="filterDate()">Filter date</button>
      </div>
      <div class="l-col-4@md"></div>
    </div>

    <table class="table table--data u-mt-20">
      <thead>
        <tr>
          <th>Icon</th>
          <th>Datetime</th>
          <th>Resource ID</th>
          <th>Code</th>
          <th>Value</th>
        </tr>
      </thead>
      <tfoot>
        <tr>
          <th>Icon</th>
          <th>Datetime</th>
          <th>Resource ID</th>
          <th>Code</th>
          <th>Value</th>
        </tr>
      </tfoot>
      <tbody>
        <template v-for="(observation, index) in this.observations">
          <tr v-if="get(['resource', 'resourceType'], observation) === 'Observation'">
            <td data-label="Icon"><i class="icon-search"></i></td>
            <td data-label="Date">{{ new Date(get(['resource', 'meta', 'lastUpdated'], observation)).toLocaleString() }}</td>
            <td data-label="Resource ID">
              <router-link :to="{ name: 'single-observation', params: { observationID: get(['resource', 'id'], observation) }}">
                {{ get(['resource', 'id'], observation) }}
              </router-link>
            </td>
            <td data-label="Code">{{ get(['resource', 'valueQuantity', 'code'], observation) }}</td>
            <td data-label="Value">{{ get(['resource', 'valueQuantity', 'value'], observation) }} {{ get(['resource', 'valueQuantity', 'unit'], observation) }}</td>
          </tr>
          <tr v-if="get(['resource', 'resourceType'], observation) === 'MedicationStatement'">
            <td data-label="Icon"><i class="icon-syringe"></i></td>
            <td data-label="Date">{{ new Date(get(['resource', 'meta', 'lastUpdated'], observation)).toLocaleString() }}</td>
            <td data-label="Resource ID">
              <router-link :to="{ name: 'single-statement', params: { statementID: get(['resource', 'id'], observation) }}">
                {{ get(['resource', 'id'], observation) }}
              </router-link>
            </td>
            <td data-label="Value">{{ get(['resource', 'medicationCodeableConcept', 'coding', 0, 'display'], observation) }} {{ get(['resource', 'valueQuantity', 'unit'], observation) }}</td>
          </tr>
        </template>

        <infinite-loading
          v-if="loadingObservations"
          v-on:infinite="infiniteHandler"
          ref="infiniteLoading"
          spinner="bubbles">
        </infinite-loading>
      </tbody>
    </table>

  </main>
</template>

<script>
  import { createNamespacedHelpers } from 'vuex'
  const { mapGetters } = createNamespacedHelpers('patient')

  export default {
    name: 'SinglePatientView',
    computed: mapGetters(['selectedPatient', 'totalVersions', 'loadingObservations', 'observations']),
    data () {
      return {
        birthDate: new Date().toISOString().split('T')[0],
        selectedDate: new Date().toISOString().split('T')[0],
        gender: 'male'
      }
    },
    mounted () {
      this.patientID = (this.$route.params.patientID).replace(/Patient\//g, '')
      this.$store.dispatch('patient/getSinglePatient', this.patientID)
      .then(data => {
        mapGetters(['selectedPatient', 'totalVersions'])
        this.birthDate = this.selectedPatient.birthDate
        this.gender = this.selectedPatient.gender
      })
    },
    methods: {
      sendUpdate () {
        this.$store.dispatch('patient/updatePatient', {
          birthDate: this.birthDate,
          patientID: this.patientID,
          gender: this.gender
        })
      },
      filterDate () {
        // var myDate = this.selectedDate
        // console.log(myDate)
        // this.observations.filter(function (record) {
        //   var date = new Date(record.resource.meta.lastUpdated)
        //   return date.getFullYear() === parseInt(myDate.split('-')[0]) &&
        //          date.getMonth() === parseInt(myDate.split('-')[1]) &&
        //          date.getDate() === parseInt(myDate.split('-')[2])
        // })
      },
      get (p, o) {
        return p.reduce((xs, x) => (xs && xs[x]) ? xs[x] : null, o)
      },
      infiniteHandler (state) {
        this.$store.dispatch('patient/getPatientObservations')
        .then(data => {
          mapGetters(['loadingObservations', 'observations'])
          if (this.loadingObservations) {
            state.loaded()
          } else {
            state.complete()
          }
        })
      }
    }
  }
</script>
