<template>
  <main class="l-app__main" role="main">

    <header class="c-toolbar">
      <div class="c-cell">
        <div class="c-cell__media">
          <i class="icon-footprint"></i>
        </div>
        <div class="c-cell__content">
          <h1 class="c-toolbar__title">Medication {{ get(['resource', 'id'], selectedMedication)}}</h1>
          <ol class="c-breadcrumb">
            <li class="c-breadcrumb__item">
              <router-link :to="{ name: 'medications' }" class="c-breadcrumb__link">Medications</router-link>
            </li>
            <li class="c-breadcrumb__item">Medication {{ get(['resource', 'id'], selectedMedication)}}</li>
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
        <tr v-if="get(['id'], selectedMedication)">
          <td data-label="Key">ID</td>
          <td data-label="Value">{{ get(['id'], selectedMedication) }}</td>
        </tr>

        <tr v-if="get(['status'], selectedMedication)">
          <td data-label="Key">Status</td>
          <td data-label="Value">{{ get(['status'], selectedMedication) }}</td>
        </tr>

        <tr v-if="get(['meta', 'versionId'], selectedMedication)">
          <td data-label="Key">Version ID</td>
          <td data-label="Value">{{ get(['meta', 'versionId'], selectedMedication) }}</td>
        </tr>

        <tr v-if="get(['meta', 'lastUpdated'], selectedMedication)">
          <td data-label="Key">Last updated</td>
          <td data-label="Value">{{ new Date(get(['meta', 'lastUpdated'], selectedMedication)).toLocaleString() }}</td>
        </tr>

        <tr v-if="get(['code', 'coding', 0, 'code'], selectedMedication)">
          <td data-label="Key">Code</td>
          <td data-label="Value">{{ get(['code', 'coding', 0, 'code'], selectedMedication) }}</td>
        </tr>

        <tr v-if="get(['code', 'coding', 0, 'display'], selectedMedication)">
          <td data-label="Key">Display</td>
          <td data-label="Value">{{ get(['code', 'coding', 0, 'display'], selectedMedication) }}</td>
        </tr>

        <tr v-if="get(['manufacturer', 'display'], selectedMedication)">
          <td data-label="Key">Manufacturer</td>
          <td data-label="Value">{{ get(['manufacturer', 'display'], selectedMedication) }}</td>
        </tr>

        <tr v-if="get(['form', 'coding', 0, 'code'], selectedMedication)">
          <td data-label="Key">Form code</td>
          <td data-label="Value">{{ get(['form', 'coding', 0, 'code'], selectedMedication) }}</td>
        </tr>

        <tr v-if="get(['form', 'coding', 0, 'display'], selectedMedication)">
          <td data-label="Key">Form display</td>
          <td data-label="Value">{{ get(['form', 'coding', 0, 'display'], selectedMedication) }}</td>
        </tr>
      </tbody>
    </table>

  </main>
</template>

<script>
  import { createNamespacedHelpers } from 'vuex'
  const { mapGetters } = createNamespacedHelpers('medication')

  export default {
    name: 'SingleMedicationView',
    computed: mapGetters(['selectedMedication']),
    mounted () {
      this.medicationID = this.$route.params.medicationID
      this.$store.dispatch('medication/getSingleMedication', this.medicationID)
      .then(data => {
        mapGetters(['selectedMedication'])
      })
    },
    methods: {
      get (p, o) {
        return p.reduce((xs, x) => (xs && xs[x]) ? xs[x] : null, o)
      }
    }
  }
</script>
